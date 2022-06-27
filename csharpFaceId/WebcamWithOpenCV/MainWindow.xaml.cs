using System;
using System.Windows;
using System.IO;
using System.Windows.Media.Imaging;
using System.Runtime.InteropServices;
using System.Diagnostics;
using Microsoft.ML;
using System.Collections.Generic;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;
using System.Linq;

namespace WebcamWithOpenCV
{
    public partial class MainWindow : System.Windows.Window
    {
       
        private WebcamStreaming _webcamStreaming;

        public MainWindow()
        {
            InitializeComponent();
            cmbCameraDevices.ItemsSource = CameraDevicesEnumerator.GetAllConnectedCameras();
            cmbCameraDevices.SelectedIndex = 0;
            cameraLoading.Visibility = Visibility.Collapsed;
        }

        private async void btnStart_Click(object sender, RoutedEventArgs e)
        {
            cameraLoading.Visibility = Visibility.Visible;
            webcamContainer.Visibility = Visibility.Hidden;
            btnStop.IsEnabled = false;
            btnStart.IsEnabled = false;

            var selectedCameraDeviceId = (cmbCameraDevices.SelectedItem as CameraDevice).OpenCvId;
            if (_webcamStreaming == null || _webcamStreaming.CameraDeviceId != selectedCameraDeviceId)
            {
                _webcamStreaming?.Dispose();
                _webcamStreaming = new WebcamStreaming(
                    imageControlForRendering: webcamPreview,
                    frameWidth: 300,
                    frameHeight: 300,
                    cameraDeviceId: cmbCameraDevices.SelectedIndex);
            }

            try
            {
                await _webcamStreaming.Start();
                btnStop.IsEnabled = true;
                btnStart.IsEnabled = false;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                btnStop.IsEnabled = false;
                btnStart.IsEnabled = true;
            }

            cameraLoading.Visibility = Visibility.Collapsed;
            webcamContainer.Visibility = Visibility.Visible;
        }

        private async void btnStop_Click(object sender, RoutedEventArgs e)
        {
            await _webcamStreaming.Stop();
            btnStop.IsEnabled = false;
            btnStart.IsEnabled = true;

            var picture = _webcamStreaming.LastPngFrame;
            string fileName = "screenshot.png";
            using (var fileStream = new FileStream(fileName, FileMode.Create))
            {
                BitmapEncoder encoder = new PngBitmapEncoder();
                var bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = new MemoryStream(picture);
                bitmapImage.EndInit();
                encoder.Frames.Add(BitmapFrame.Create(bitmapImage));
                encoder.Save(fileStream);
            }
            string text = _webcamStreaming.AnalyzeImage();
            finalResults.Text = text;
            _webcamStreaming?.Dispose();
        }

        private void btnUpload_Click(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Please add your Pictures, with the name of the person into the TrainData folder!");
            ProcessStartInfo startinfo = new ProcessStartInfo();
            startinfo.FileName = $"{Directory.GetCurrentDirectory()}/../../../../FaceIdTrain.ConsoleApp/bin/Debug/net6.0/FaceIdTrain.ConsoleApp.exe";
            startinfo.CreateNoWindow = true;
            startinfo.UseShellExecute = true;

            Process.Start($"{Directory.GetCurrentDirectory()}/../../../../../TrainData");
            Process p = Process.Start(startinfo);
            p.WaitForExit();

            MessageBox.Show("Model Creation finished!");
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            _webcamStreaming?.Dispose();
        }
    }
}
