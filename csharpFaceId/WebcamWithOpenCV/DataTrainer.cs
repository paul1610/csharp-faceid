using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using WebcamWithOpenCV;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

namespace ImageClassification.Train
{
    public class DataTrainer
    {
        /*public static void TrainModel(string imagePath)
        {
            string outputMlNetModelFilePath = "FaceModel.zip";

            var mlContext = new MLContext(seed: 1);

            // 1. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
            IEnumerable<InMemoryImageData> images = LoadImagesFromDirectory(folder: imagePath);
            IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledFullImageFilePathsDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

            // 2. Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical)
            IDataView shuffledFullImagesDataset = mlContext.Transforms.Conversion.
                    MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.LoadRawImageBytes(
                                                outputColumnName: "Image",
                                                imageFolder: imagePath,
                                                inputColumnName: "ImageFileName"))
                .Fit(shuffledFullImageFilePathsDataset)
                .Transform(shuffledFullImageFilePathsDataset);

            // 3. Define the model's training pipeline
            var pipeline = mlContext.MulticlassClassification.Trainers
                    .ImageClassification(featureColumnName: @"Image",
                                         labelColumnName: "LabelAsKey")
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel",
                                                                      inputColumnName: @"PredictedLabel"));

            // 4. Train/create the ML model
            // Measuring training time
            var watch = Stopwatch.StartNew();

            //Train
            ITransformer trainedModel = pipeline.Fit(shuffledFullImagesDataset);
            
            var elapsedMs = watch.ElapsedMilliseconds;

            // 5. Save the model to assets/outputs (ML.NET .zip model file and TensorFlow .pb model file)
            mlContext.Model.Save(trainedModel, shuffledFullImagesDataset.Schema, outputMlNetModelFilePath);
            MessageBox.Show($"Training took: {elapsedMs / 1000} seconds");
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
    }*/

        public static void TrainModel(string imagePath)
        {
            string outputMlNetModelFilePath = "FaceModel.zip";

            var mlContext = new MLContext(seed: 1);

            // 2. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
            IEnumerable<InMemoryImageData> images = LoadImagesFromDirectory(folder: imagePath);
            IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledFullImageFilePathsDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

            // 3. Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical)
            IDataView shuffledFullImagesDataset = mlContext.Transforms.Conversion.
                    MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.LoadRawImageBytes(
                                                outputColumnName: "Image",
                                                imageFolder: imagePath,
                                                inputColumnName: "ImageFileName"))
                .Fit(shuffledFullImageFilePathsDataset)
                .Transform(shuffledFullImageFilePathsDataset);

            // 4. Split the data 80:20 into train and test sets, train and evaluate.
            var trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2);
            IDataView trainDataView = trainTestData.TrainSet;
            IDataView testDataView = trainTestData.TestSet;

            // 5. Define the model's training pipeline using DNN default values
            //
            var pipeline = mlContext.MulticlassClassification.Trainers
                    .ImageClassification(featureColumnName: "Image",
                                         labelColumnName: "LabelAsKey",
                                         validationSet: testDataView)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel",
                                                                      inputColumnName: "PredictedLabel"));

            // 6. Train/create the ML model

            // Measuring training time
            var watch = Stopwatch.StartNew();

            //Train
            ITransformer trainedModel = pipeline.Fit(trainDataView);

            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;

            MessageBox.Show($"Training with transfer learning took: {elapsedMs / 1000} seconds");

            // 8. Save the model to assets/outputs (You get ML.NET .zip model file and TensorFlow .pb model file)
            mlContext.Model.Save(trainedModel, trainDataView.Schema, outputMlNetModelFilePath);
            mlContext.Model.Save(trainedModel, trainDataView.Schema, "C:/Users/nellp/Desktop/Schule/2DHIF - 2021-22/PR0 OO/Übungen/08_Project/csharp-faceid/csharp-faceid/csharpFaceId/WebcamWithOpenCV/FaceModel.zip");
            MessageBox.Show($"Model saved to: {outputMlNetModelFilePath}");
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
    }
}

