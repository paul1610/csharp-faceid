using ImageProcessor;
using ImageProcessor.Imaging;
using Microsoft.ML.Data;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Controls;
using Microsoft.ML;
using System.Windows;
using System.Windows.Documents;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;
using System.Linq;

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
        public string AnalyzeImage()
        {
            // Make a single prediction on the sample data and print results
            var predictionResult = FaceModel.Predict();

            string final = $"Final Result: {predictionResult.PredictedLabel}\n";
            string[] names = Directory.GetDirectories("../../../../../TrainData");
            float recognized = 0;

            for (int i = 0; i < names.Length; i++)
            {
                names[i] = names[i].Split('\\').Last();
                if (predictionResult.Score[i] > recognized)
                {
                    recognized = predictionResult.Score[i];
                }

                final += $"\nPerson: {names[i]} \n Score: {predictionResult.Score[i] * 100}%\n";
            }

            if (recognized >= 0.6)
            {
                return final;
            }

            return "No Face was detected! \nPlease try again...";
        }
        public partial class FaceModel
        {
            private static string MLNetModelPath = Path.GetFullPath("FaceModel.zip");

            public static readonly Lazy<PredictionEngine<InMemoryImageData, ImagePrediction>> PredictEngine = new Lazy<PredictionEngine<InMemoryImageData, ImagePrediction>>(() => CreatePredictEngine(), true);

            /// <summary>
            /// Use this method to predict on <see cref="ModelInput"/>.
            /// </summary>
            /// <param name="input">model input.</param>
            /// <returns><seealso cref=" ModelOutput"/></returns>
            public static ImagePrediction Predict()
            {
                try
                {
                    var mlContext = new MLContext(seed: 1);
                    ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);

                    // Create prediction function to try one prediction
                    var predictionEngine = mlContext.Model
                        .CreatePredictionEngine<InMemoryImageData, ImagePrediction>(mlModel);

                    IEnumerable<InMemoryImageData> testImages = LoadImagesFromDirectory("./");

                    var temp = testImages.First();
                    var imageToPredict = testImages.First();

                    var prediction = predictionEngine.Predict(imageToPredict);

                    /*MessageBox.Show(
                        $"Image Filename : [{imageToPredict.Label}], " +
                        $"Scores : [{string.Join(",", prediction.Score)}], " +
                        $"Predicted Label : {prediction.PredictedLabel}");*/
                    return prediction;
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
                return null;
            }

            public static IEnumerable<InMemoryImageData> LoadImagesFromDirectory(string folder)
            {
                var files = Directory.GetFiles(folder, "*",
                              searchOption: SearchOption.AllDirectories);

                foreach (var file in files)
                {
                    if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                        continue;

                    var temp = Path.GetDirectoryName(file);
                    var label = temp.Split('\\')[temp.Split('\\').Length - 1];

                    yield return new InMemoryImageData(File.ReadAllBytes(file), label, label);
                }
            }

            private static PredictionEngine<InMemoryImageData, ImagePrediction> CreatePredictEngine()
            {
                var mlContext = new MLContext();
                ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
                return mlContext.Model.CreatePredictionEngine<InMemoryImageData, ImagePrediction>(mlModel);
            }
        }
        public void Dispose()
        {
            _cancellationTokenSource?.Cancel();
            _lastFrame?.Dispose();
        }
    }
}