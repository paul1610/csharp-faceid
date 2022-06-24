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
            FaceModel.ModelInput sampleData = new FaceModel.ModelInput()
            {
                ImageSource = Path.GetFullPath("screenshot.png")
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = FaceModel.Predict(sampleData);

            string final = $"Final Result: {predictionResult.Prediction}\n";
            string[] names = { "Jonas", "Niklas", "Paul", "Test" };
            float recognized = 0;

            for (int i = 0; i < names.Length; i++)
            {
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
                catch (Exception ex)
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

        // -----


        private static string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        //private static string _imagesFolder = Path.Combine(_assetsPath, "images");
        private static string _imagesFolder = "C:/Users/nellp/Desktop/Schule/2DHIF - 2021-22/PR0 OO/Übungen/08_Project/csharp-faceid/csharp-faceid/";
        private static string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
        private static string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
        private static string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
        private static string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

        struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            var model = GenerateModel(mlContext);
        }

        private static ITransformer GenerateModel(MLContext mlContext)
        {
            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                // The image transforms transform the images into the model's expected format.
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).
                    ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            IEnumerable<ImageData> temp = LoadImagesFromDirectory(folder: _imagesFolder, useFolderNameAsLabel: true);
            IDataView trainingData = mlContext.Data.LoadFromEnumerable(temp);

            ITransformer createdModel = pipeline.Fit(trainingData);

            mlContext.Model.Save(createdModel, trainingData.Schema, "FaceModel.zip");
            return createdModel;
        }

        void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
            }
        }



        /* public static void Run()
         {
             //path definition & initialisation
             //var projectDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
             var projectDir = "C:/Users/nellp/Desktop/Schule/2DHIF - 2021-22/PR0 OO/Übungen/08_Project/csharp-faceid/csharp-faceid/";
             var assets = Path.Combine(projectDir, "TestData/Test");

             //new ml context
             MLContext mlContext = new MLContext();

             //get the list of images
             IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: projectDir, useFolderNameAsLabel: true);
             IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

             //data preprocessing
             IDataView shuffle = mlContext.Data.ShuffleRows(imgData);

            //set format
            // var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey
                 //takes the categorical value in the Label column, convert it to a numerical KeyType value and store it in a new column called LabelKey
                 //(inputColumnName: "Label",
                 //outputColumnName: "LabelKey")
                 //take the values from the ImgPath column along with the imageFolder 
                 //parameter to load images for training the model
                 //.Append(mlContext.Transforms.LoadRawImageBytes(
                 //outputColumnName: "Img",
                 //imageFolder: assets,
                 //inputColumnName: "ImgPath"));

             var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(@"Label", @"Label")
                                     .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: @"Img", imageFolder: assets, inputColumnName: @"ImgPath"))
                                     .Append(mlContext.Transforms.CopyColumns(@"Features", @"Img"))
                                     .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(labelColumnName: @"Label"))
                                     .Append(mlContext.Transforms.Conversion.MapKeyToValue(@"PredictedLabel", @"PredictedLabel"));

             IDataView preProcData = preprocessingPipeline.Fit(shuffle).Transform(shuffle);

             //create traindata
             TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcData, testFraction: 0.3);
             TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

             //create idataview for each split
             IDataView trainSet = trainSplit.TrainSet;
             IDataView validationSet = validationTestSplit.TrainSet;
             IDataView testSet = validationTestSplit.TestSet;

             //define model train pipeline
             var classifierOptions = new ImageClassificationTrainer.Options()
             {
                 //input column for the model
                 FeatureColumnName = "Img",
                 //target variable column 
                 LabelColumnName = "Label",
                 //IDataView containing validation set
                 ValidationSet = validationSet,
                 //define pretrained model to be used
                 Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                 //track progress during model training
                 MetricsCallback = (metrics) => Console.WriteLine(metrics),
                 //if TestOnTrainSet is set to true, model is evaluated against 
                   //Training set if validation set is not there
                 TestOnTrainSet = false,
                 //whether to use cached bottleneck values in further runs
                 ReuseTrainSetBottleneckCachedValues = true,
                 //similar to ReuseTrainSetBottleneckCachedValues but for validation 
                   //set instead of train set
                 ReuseValidationSetBottleneckCachedValues = true
             };

             var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions);

             //fit the data into the training pipeline
             ITransformer trainedModel = trainingPipeline.Fit(trainSet);

             mlContext.Model.Save(trainedModel, testSet.Schema, "FaceModel.zip");

             //classify
             // ClassifyMultiple(mlContext, testSet, trainedModel);
         }

     

         public static void ClassifyMultiple(MLContext mlContext, IDataView data, ITransformer trainedModel)
         {
             IDataView predictionData = trainedModel.Transform(data);
             IEnumerable<Output> predictions = mlContext.Data.CreateEnumerable<Output>(predictionData, reuseRowObject: true).Take(20);

             MessageBox.Show("Prediction for multiple images");
             MessageBox.Show(predictionData.ToString());

             foreach (var p in predictions)
             {
                 MessageBox.Show(p.Label);
                 string imgName = Path.GetFileName(p.ImgPath);
                 MessageBox.Show($"Image: {imgName} | Actual Label: {p.Label} | Predicted Label: {p}");
             }
         }
         */
        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            //get all file paths from the subdirectories
            var files = Directory.GetFiles(folder, "*", searchOption:
            SearchOption.AllDirectories);

            //iterate through each file
            foreach (var file in files)
            {
                //Image Classification API supports .jpg and .png formats; check img formats
                if ((Path.GetExtension(file) != ".jpg") &&
                 (Path.GetExtension(file) != ".png"))
                    continue;

                //store filename in a variable, say ‘label’
                var label = Path.GetFileName(file);

                //If the useFolderNameAsLabel parameter is set to true, then name 
                // of parent directory of the image file is used as the label. Else label is expected to be the file name or a a prefix of the file name. 
                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                //create a new instance of ImgData()
                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }

        // -----

    }
}