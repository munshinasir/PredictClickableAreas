import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvKNearest;
import org.opencv.objdetect.HOGDescriptor;

public class KnnProcessorSampleImg {

	public static void main(String[] args) {

		// Loading the native library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		// setting the training samples
		Mat trainingData = new Mat(16, 9, CvType.CV_32FC1);
		trainingData.setTo(new Scalar(0.0));
		Mat trainingClasses = new Mat(16, 1, CvType.CV_32FC1);
		trainingClasses.setTo(new Scalar(0.0));

		for (int i = 0; i < 4; i++) {

			Mat inputImage = Highgui
					.imread("/home/nasir/Documents/Projects/OpenCV/Workspace/RobotDemo2/Images/Blot"
							+ (i + 1) + ".png");

			Mat grayImage = new Mat();

			// converting the image into grayscale
			Imgproc.cvtColor(inputImage, grayImage, Imgproc.COLOR_BGR2GRAY);

			Mat sample = new Mat();
			// extracting a 16X16 part of the image
			// sample = new Mat(grayImage, new Rect(500, 250, 128, 256));

			// trying on resizing the image
			Imgproc.resize(grayImage, sample, new Size(16, 16));

			// creating the hog descriptor for the sample
			MatOfFloat ders = computeHog(sample);
			prepareData(ders, trainingData, i);
			// need to copy the entire image into the columns
			/*
			 * int row = 4 * i; int j = 0; for (int k = row; k < (row + 4); k++)
			 * {
			 * 
			 * for (int col1 = 0; col1 < 9; col1++) {
			 * 
			 * trainingData.rowRange(k, k + 1).colRange(col1, col1 + 1)
			 * .setTo(ders.rowRange(j, j + 1)); j++; } }
			 */

		}

		// setting up the testing classes
		trainingClasses.rowRange(0, 7).setTo(new Scalar(1.0));
		trainingClasses.rowRange(8, 15).setTo(new Scalar(2.0));

		CvKNearest knn = new CvKNearest(trainingData, trainingClasses);

		Mat testImage = Highgui
				.imread("/home/nasir/Documents/Projects/OpenCV/Files/lena1.png");

		//
		Mat grayImage = new Mat();

		// converting the image into grayscale
		Imgproc.cvtColor(testImage, grayImage, Imgproc.COLOR_BGR2GRAY);

		Mat sample = new Mat();
		// extracting a 16X16 part of the image
		// sample = new Mat(grayImage, new Rect(500, 250, 128, 256));

		// trying on resizing the image
		Imgproc.resize(grayImage, sample, new Size(16, 16));

		// computing the hog descriptors
		MatOfFloat ders = computeHog(sample);

		Mat testData = new Mat(ders.rows() / 9, 9, CvType.CV_32FC1);
		prepareData(ders, testData, 0);

		System.out.println(ders.rows() / 9);
		Mat results = new Mat(ders.rows() / 9, 1, CvType.CV_32FC1);
		Mat neighborResponses = new Mat(testData.size(), CvType.CV_32FC1);
		Mat dists = new Mat(testData.size(), CvType.CV_32FC1);

		knn.find_nearest(testData, 5, results, neighborResponses, dists);

		System.out.println("Results------------------------------------");
		System.out.println(results.dump());
	}

	public static MatOfFloat computeHog(Mat image) {

		MatOfFloat ders = new MatOfFloat();
		MatOfPoint locs = new MatOfPoint();

		// creating the hog descriptor for the sample
		HOGDescriptor hog = new HOGDescriptor(new Size(16, 16),
				new Size(16, 16), new Size(8, 8), new Size(8, 8), 9);

		// computing the descriptors for the sample
		hog.compute(image, ders, new Size(0, 0), new Size(0, 0), locs);

		return ders;

	}

	public static void prepareData(MatOfFloat ders, Mat data, int i) {

		int row = 4 * i;
		int j = 0;
		for (int k = row; k < (row + 4); k++) {

			for (int col1 = 0; col1 < 9; col1++) {

				data.rowRange(k, k + 1).colRange(col1, col1 + 1)
						.setTo(ders.rowRange(j, j + 1));
				j++;
			}
		}

	}

}
