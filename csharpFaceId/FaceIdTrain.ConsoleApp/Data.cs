using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WebcamWithOpenCV
{
    public class ImageData
    {
        public ImageData(string imageSource, string label)
        {
            ImageSource = imageSource;
            Label = label;
        }

        public readonly string ImageSource;

        public readonly string Label;
    }

    public class InMemoryImageData
    {
        public InMemoryImageData(byte[] image, string label, string imageFileName)
        {
            Image = image;
            Label = label;
            ImageFileName = imageFileName;
        }

        public readonly byte[] Image;

        public readonly string Label;

        public readonly string ImageFileName;
    }

    public class ImagePrediction
    {
        [ColumnName("Score")]
        public float[] Score;

        [ColumnName("PredictedLabel")]
        public string PredictedLabel;
    }
}