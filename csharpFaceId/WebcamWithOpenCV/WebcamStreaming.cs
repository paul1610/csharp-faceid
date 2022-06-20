using ImageProcessor;
using ImageProcessor.Imaging;
using Microsoft.ML.Data;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Controls;
using Microsoft.ML;
using System.Windows;

namespace WebcamWithOpenCV
{
    public sealed class WebcamStreaming : IDisposable
    {
        private System.Drawing.Bitmap _lastFrame;
        private Task _previewTask;

        private CancellationTokenSource _cancellationTokenSource;
        private readonly Image _imageControlForRendering;
        private readonly int _frameWidth;
        private readonly int _frameHeight;

        public int CameraDeviceId { get; private set; }
        public byte[] LastPngFrame { get; private set; }
        public bool FlipHorizontally = true;

        public WebcamStreaming(
            Image imageControlForRendering,
            int frameWidth,
            int frameHeight,
            int cameraDeviceId)
        {
            _imageControlForRendering = imageControlForRendering;
            _frameWidth = frameWidth;
            _frameHeight = frameHeight;
            CameraDeviceId = cameraDeviceId;
        }

        public async Task Start()
        {
            // Never run two parallel tasks for the webcam streaming
            if (_previewTask != null && !_previewTask.IsCompleted)
                return;

            var initializationSemaphore = new SemaphoreSlim(0, 1);

            _cancellationTokenSource = new CancellationTokenSource();
            _previewTask = Task.Run(async () =>
            {
                try
                {
                    // Creation and disposal of this object should be done in the same thread 
                    // because if not it throws disconnectedContext exception
                    var videoCapture = new VideoCapture();

                    if (!videoCapture.Open(CameraDeviceId))
                    {
                        throw new ApplicationException("Cannot connect to camera");
                    }

                    using (var frame = new Mat())
                    {
                        while (!_cancellationTokenSource.IsCancellationRequested)
                        {
                            videoCapture.Read(frame);

                            if (!frame.Empty())
                            {

                                // Releases the lock on first not empty frame
                                if (initializationSemaphore != null)
                                    initializationSemaphore.Release();

                                _lastFrame = FlipHorizontally 
                                    ? BitmapConverter.ToBitmap(frame.Flip(FlipMode.Y))
                                    : BitmapConverter.ToBitmap(frame);

                                var lastFrameBitmapImage = _lastFrame.ToBitmapSource();
                                lastFrameBitmapImage.Freeze();
                                _imageControlForRendering.Dispatcher.Invoke(
                                    () => _imageControlForRendering.Source = lastFrameBitmapImage);
                            }

                            // 30 FPS
                            await Task.Delay(33);
                        }
                    }

                    videoCapture?.Dispose();
                }
                finally
                {
                    if (initializationSemaphore != null)
                        initializationSemaphore.Release();
                }

            }, _cancellationTokenSource.Token);

            // Async initialization to have the possibility to show an animated loader without freezing the GUI
            // The alternative was the long polling. (while !variable) await Task.Delay
            await initializationSemaphore.WaitAsync();
            initializationSemaphore.Dispose();
            initializationSemaphore = null;

            if (_previewTask.IsFaulted)
            {
                // To let the exceptions exit
                await _previewTask;
            }
        }

        public async Task Stop()
        {
            // If "Dispose" gets called before Stop
            if (_cancellationTokenSource.IsCancellationRequested)
                return;

            if (!_previewTask.IsCompleted)
            {
                _cancellationTokenSource.Cancel();

                // Wait for it, to avoid conflicts with read/write of _lastFrame
                await _previewTask;
            }

            if (_lastFrame != null)
            {
                using (var imageFactory = new ImageFactory())
                using (var stream = new MemoryStream())
                {
                    imageFactory
                        .Load(_lastFrame)
                        .Resize(new ResizeLayer(
                            size: new System.Drawing.Size(_frameWidth, _frameHeight),
                            resizeMode: ImageProcessor.Imaging.ResizeMode.Crop,
                            anchorPosition: AnchorPosition.Center))
                        .Save(stream);
                    LastPngFrame = stream.ToArray();
                }
            }
            else
            {
                LastPngFrame = null;
            }
        }
        public void AnalyzeImage()
        {
            FaceModel.ModelInput sampleData = new FaceModel.ModelInput()
            {
                ImageSource = Path.GetFullPath("screenshot.png")
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = FaceModel.Predict(sampleData);
            MessageBox.Show($"\n\nPredicted Label value: {predictionResult.Prediction} \nPredicted Label scores: [{String.Join(",", predictionResult.Score)}]\n\n");

        }
        public partial class FaceModel
        {
            /// <summary>
            /// model input class for FaceModel.
            /// </summary>
            #region model input class
            public class ModelInput
            {
                [ColumnName(@"Label")]
                public string Label { get; set; }

                [ColumnName(@"ImageSource")]
                public string ImageSource { get; set; }

            }

            #endregion

            /// <summary>
            /// model output class for FaceModel.
            /// </summary>
            #region model output class
            public class ModelOutput
            {
                [ColumnName("PredictedLabel")]
                public string Prediction { get; set; }

                public float[] Score { get; set; }
            }

            #endregion

            private static string MLNetModelPath = Path.GetFullPath("FaceModel.zip");

            public static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(() => CreatePredictEngine(), true);

            /// <summary>
            /// Use this method to predict on <see cref="ModelInput"/>.
            /// </summary>
            /// <param name="input">model input.</param>
            /// <returns><seealso cref=" ModelOutput"/></returns>
            public static ModelOutput Predict(ModelInput input)
            {
                try
                {
                    var predEngine = PredictEngine.Value;
                    return predEngine.Predict(input);
                }
                catch(Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
                return null;
            }

            private static PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
            {
                var mlContext = new MLContext();
                ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
                return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
            }
        }
        public void Dispose()
        {
            _cancellationTokenSource?.Cancel();
            _lastFrame?.Dispose();
        }

    }
}
