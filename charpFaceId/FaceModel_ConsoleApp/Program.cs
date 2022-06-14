﻿
// This file was auto-generated by ML.NET Model Builder. 

using System;

namespace FaceModel_ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = PathSettings.TestImageFolerPath + @"\Jonas\jonas1.jpg";
            // Create single instance of sample data from first line of dataset for model input
            FaceModel.ModelInput sampleData = new FaceModel.ModelInput()
            {
                ImageSource = path,
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = FaceModel.Predict(sampleData);

            Console.WriteLine("Using model to make single prediction -- Comparing actual Label with predicted Label from sample data...\n\n");


            Console.WriteLine($"ImageSource: {path}");


            Console.WriteLine($"\n\nPredicted Label value: {predictionResult.Prediction} \nPredicted Label scores: [{String.Join(",", predictionResult.Score)}]\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}