using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WebcamWithOpenCV
{
    /*   public class ImgData
       {
           //path of the image file
           public string ImgPath { get; set; }
           //category to which the image in ImgPath belongs to    
           public string Label { get; set; }
       }
       public class InputData
       {
           public byte[] Img { get; set; } //byte representation of image
           public UInt32 LabelKey { get; set; } //numerical representation of label
           public string ImgPath { get; set; } //path of the image
           public string Label { get; set; }
       }

       public class Output
       {
           public string ImgPath { get; set; } //path of the image
           public string Label { get; set; } //target category
           public UInt32 PredictedLabel { get; set; } //predicted label
       }*/

    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }

    public class ImagePrediction : ImageData
    {
        public float[] Score;

        public string PredictedLabelValue;
    }


}
