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
using System.Threading;

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

            var fileName = "screenshot.png";
            var picture = _webcamStreaming.LastPngFrame;
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
            webcamContainer.Visibility = Visibility.Collapsed;
        }

        private async void btnUpload_Click(object sender, RoutedEventArgs e)
        {
            string dirName = InputTextBox.Text;
            if (dirName != "" && dirName.Contains(" "))
                MessageBox.Show("Please input a valid Name!");
            else
            {
                int count = 0;
                string newDir = $"{Directory.GetCurrentDirectory()}/../../../../../TrainData/{dirName}";
                if(Directory.Exists(newDir))
                {
                    string[] files = Directory.GetFiles(newDir);
                    foreach (string file in files)
                    {
                        File.SetAttributes(file, FileAttributes.Normal);
                        File.Delete(file);
                    }

                    Directory.Delete(newDir);
                }
                Directory.CreateDirectory(newDir);

                btnStart_Click(null, null);
                MessageBox.Show("Starting making Images!");

                var fileName = "modelCreation";
                do
                {
                    using (var fileStream = new FileStream($"{fileName}{count}.png", FileMode.Create))
                    {
                        await _webcamStreaming.Stop();
                        var picture = _webcamStreaming.LastPngFrame;

                        BitmapEncoder encoder = new PngBitmapEncoder();
                        var bitmapImage = new BitmapImage();
                        bitmapImage.BeginInit();
                        bitmapImage.StreamSource = new MemoryStream(picture);
                        bitmapImage.EndInit();
                        encoder.Frames.Add(BitmapFrame.Create(bitmapImage));
                        encoder.Save(fileStream);

                        Thread.Sleep(500);
                        count++;
                        await _webcamStreaming.Start();
                    }
                } while (count <= 20);

                for(int i = 0; i <= 20; i++)
                {
                    File.Move($"{fileName}{i}.png", $"{Directory.GetCurrentDirectory()}/../../../../../TrainData/{dirName}/{dirName}{i}.png");
                }

                MessageBox.Show("Finished making Images!");

                _webcamStreaming?.Dispose();
                webcamContainer.Visibility = Visibility.Collapsed;

                ProcessStartInfo startinfo = new ProcessStartInfo();
                startinfo.FileName = $"{Directory.GetCurrentDirectory()}/../../../../FaceIdTrain.ConsoleApp/bin/Debug/net6.0/FaceIdTrain.ConsoleApp.exe";
                startinfo.CreateNoWindow = true;
                startinfo.UseShellExecute = true;
                Process p = Process.Start(startinfo);
                p.WaitForExit();

                MessageBox.Show("Model Creation finished!");
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            _webcamStreaming?.Dispose();
        }
    }
}
