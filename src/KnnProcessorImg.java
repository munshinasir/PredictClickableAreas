import java.awt.Rectangle;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

import net.sourceforge.tess4j.Tesseract1;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Range;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvKNearest;
import org.opencv.objdetect.HOGDescriptor;

public class KnnProcessorImg {

	Size cellSize;
	Mat trainingClasses;
	Mat trainingData;
	float[] cellAttributeArray;
	int[] tessAttributeArray;
	int cellIndex;
	int tessIndex;
	CvKNearest knn;

	public KnnProcessorImg() {

		// initializing variables
		cellSize = new Size(8, 8);

		// setting the training samples and classes matrices
		//trainingClasses = new Mat(1, 1, CvType.CV_32FC1);
		trainingData = new Mat(1, 9, CvType.CV_32FC1);

		trainingData.setTo(new Scalar(0.0));
		//trainingClasses.setTo(new Scalar(0.0));

		 cellAttributeArray = new float[130560];
		//cellAttributeArray = new float[8065];
		cellIndex = 0;
		/*
		 * tessAttributeArray = new int[130560]; tessIndex = 0;
		 */

	}

	public static void main(String[] args) {

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		// Creating the object of the knn Processor
		KnnProcessorImg knnProc = new KnnProcessorImg();

		// Training Data
		knnProc.trainData();

		// Testing Data and displaying the results
		knnProc.testData();

	}

	public void trainData() {
		// fetching the folder for the data
		File folderName = new File("data");

		// assuming that there are exactly the same number of files in both
		// folders
		int numberOfFiles = folderName.listFiles()[0].listFiles().length;
		int prevNumberOfCells = 0;

		for (int imageNo = 0; imageNo < numberOfFiles; imageNo++) {

			// read the image
			Mat inputImage = Highgui.imread("data/Images/"
					+ folderName.listFiles()[0].listFiles()[imageNo].getName());

			// read the XML Annotations
			XmlReader xml = new XmlReader();
			xml.readAnnotations(folderName.listFiles()[1].listFiles()[imageNo]);
			int[] attributeArray = xml.getAttributeArray();
			int[][] pointsArray = xml.getPointsArray();

			// processing the image
			Mat grayImage = new Mat();
			Mat sample = new Mat();

			// converting the image into grayscale
			Imgproc.cvtColor(inputImage, grayImage, Imgproc.COLOR_BGR2GRAY);

			// Resizing the image to accomodate the block size of 8X8
			// Imgproc.resize(grayImage, sample, new
			// Size((grayImage.width()/cellSize.width)*cellSize.width ,
			// (grayImage.height()/cellSize.height)*cellSize.height));
			Imgproc.resize(grayImage, sample, new Size(
					(grayImage.width() / 8) * 8,
					(grayImage.height() / 8) * 8));

			// preparing the training Data
			int cellsInX = (int) (sample.size().width / cellSize.width);
			int cellsInY = (int) (sample.size().height / cellSize.height);
			int totalnumberOfCells = cellsInX * cellsInY;

			// creating the hog descriptor for the sample
			MatOfFloat ders = computeHog(sample);
			// computing the average of the descriptors
			HOGVizualizer hogV = new HOGVizualizer();
			Mat avgDers = hogV.getHogDescriptorVisualImage(sample, ders,
					sample.size(), cellSize);

			prevNumberOfCells = prepareData(avgDers, trainingData,
					prevNumberOfCells, totalnumberOfCells);

			/*
			 * Annotating the classes : Finding where each polygon belongs and
			 * adding that into the training classes mat
			 */

			Tesseract1 instance = new Tesseract1(); // JNA Direct Mapping

			// slotting the image into smaller cells
			for (int recty = 0; recty <= ((int) sample.size().height - 8); recty = recty + 8) {

				for (int rectx = 0; rectx <= ((int) sample.size().width - 8); rectx = rectx + 8) {

					// Checking if rectx, recty, rectx+8, recty+8 belongs to
					// a polygon
					// adding that to the attribute class
					cellAttributeArray[cellIndex] = getAttributeClass(rectx,
							recty, rectx + 8, recty + 8, pointsArray,
							attributeArray);
					cellIndex++;

					// System.out.println("Cell indexing done");

					/*
					 * At present Tesseract is not working with small blocks
					 * like the ones that I have
					 */

					/*
					 * File imageFile = new File("data/Images/"+
					 * folderName.listFiles
					 * ()[0].listFiles()[imageNo].getName()); String str =
					 * instance.doOCR(imageFile, new Rectangle(rectx, recty, 8,
					 * 8)); // adding the tess attribute if there is some
					 * string in the cell if (str == null || (str.length() ==
					 * 0)) { tessAttributeArray[tessIndex]= 0;
					 * //System.out.println("Tess : " +
					 * tessAttributeArray[tessIndex]); tessIndex++; } else {
					 * tessAttributeArray[tessIndex] = 1;
					 * //System.out.println("Tess : " +
					 * tessAttributeArray[tessIndex] + " - String : " +
					 * str.length() + " :  " + str); tessIndex++; }
					 */
					// System.out.println("String indexing done");

				}

			}

		}

		try {

			File file = new File(
					"/home/nasir/Documents/Projects/OpenCV/Workspace/IconPredictionTrial1/output/trainingData-5.txt");
			PrintWriter pw = new PrintWriter(file);
			pw.println(trainingData.dump());

			// converting the training Classes to Mat
			trainingClasses = toMat(cellAttributeArray);

			File file1 = new File(
					"/home/nasir/Documents/Projects/OpenCV/Workspace/IconPredictionTrial1/output/trainingClasses-5.txt");
			PrintWriter pw1;
			pw1 = new PrintWriter(file1);
			for (int i = 0; i < cellAttributeArray.length; i++) {
				pw1.println(cellAttributeArray[i]);
			}

		} catch (Exception e) {

			e.printStackTrace();
		}

		System.out.println(trainingData.rows() +" - "+ trainingClasses.rows() + "-" + cellAttributeArray.length);

		// train the classifier
		knn = new CvKNearest(trainingData, trainingClasses);

		System.out.println("Average Rows in training data: "
				+ trainingData.rows());
		System.out.println("Rows in attributes : " + cellAttributeArray.length);

	}

	public void testData() {

		Mat testImage = Highgui
				.imread("/home/nasir/Documents/Projects/OpenCV/Workspace/IconPredictionTrial1/test/Images/sample_image.jpg");

		// reading the xml annotations
		XmlReader xml = new XmlReader();
		xml.readAnnotations(new File(
				"/home/nasir/Documents/Projects/OpenCV/Workspace/IconPredictionTrial1/test/Annotations/sample_image.xml"));
		int[] attributeArrayTest = xml.getAttributeArray();
		int[][] pointsArrayTest = xml.getPointsArray();

		Mat grayTestImage = new Mat();

		// converting the image into grayscale
		Imgproc.cvtColor(testImage, grayTestImage, Imgproc.COLOR_BGR2GRAY);

		Mat sampleTest = new Mat();
		Imgproc.resize(
				grayTestImage,
				sampleTest,
				new Size((grayTestImage.width() / 8) * 8, (grayTestImage
						.height() / 8) * 8));

		// preparing the training Data
		int cellsInX = (int) (sampleTest.size().width / cellSize.width);
		int cellsInY = (int) (sampleTest.size().height / cellSize.height);
		int totalnumberOfCells = cellsInX * cellsInY;

		// creating the hog descriptor for the sample
		MatOfFloat ders = computeHog(sampleTest);
		// computing the average of the descriptors
		HOGVizualizer hogV = new HOGVizualizer();
		Mat avgDers = hogV.getHogDescriptorVisualImage(sampleTest, ders,
				sampleTest.size(), cellSize);

		// printing to the file
		try {
			File file = new File(
					"/home/nasir/Documents/Projects/OpenCV/Workspace/IconPredictionTrial1/output/testdata-5.txt");
			PrintWriter pw = new PrintWriter(file);
			pw.println(avgDers.dump());
		} catch (Exception e) {
			e.printStackTrace();
		}

		int prevNumberOfCells = 0;
		Mat testData = new Mat(1, 9, CvType.CV_32FC1);
		prevNumberOfCells = prepareData(avgDers, testData, prevNumberOfCells,
				totalnumberOfCells);

		float[] cellAttributeArrayTest = new float[testData.rows()];
		int cellIndexTest = 0;

		// slotting the image and getting the attribute values
		for (int recty = 0; recty <= ((int) sampleTest.size().height - 8); recty = recty + 8) {

			for (int rectx = 0; rectx <= ((int) sampleTest.size().width - 8); rectx = rectx + 8) {

				cellAttributeArrayTest[cellIndexTest] = getAttributeClass(
						rectx, recty, rectx + 8, recty + 8, pointsArrayTest,
						attributeArrayTest);
				cellIndexTest++;
			}
		}

		Mat results = new Mat(avgDers.rows(), 1, CvType.CV_32FC1);
		Mat neighborResponses = new Mat(testData.size(), CvType.CV_32FC1);
		Mat dists = new Mat(testData.size(), CvType.CV_32FC1);

		knn.find_nearest(testData, 3, results, neighborResponses, dists);

		System.out.println(results.dump());

		// printing the classes to file
		try {
			/*
			 * File file1 = new File(
			 * "/home/nasir/Documents/Projects/OpenCV/Workspace/IconPredictionTrial1/output/testClasses-2.txt"
			 * ); PrintWriter pw1 = new PrintWriter(file1);
			 * 
			 * for(int i=0; i < results.rows() ; i++)
			 * pw1.println(results.rowRange(i, i+1).dump());
			 */

			File file2 = new File(
					"/home/nasir/Documents/Projects/OpenCV/Workspace/IconPredictionTrial1/output/testClasses-5.txt");
			PrintWriter pw2 = new PrintWriter(file2);

			for (int i = 0; i < cellAttributeArrayTest.length; i++) {
				pw2.println(cellAttributeArrayTest[i]);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		System.out.println("Average Rows in the test data: " + avgDers.rows()
				+ " Test data : " + testData.rows());
		System.out.println("Rows in resutls : " + results.rows());
		System.out.println("Rows in attributes : "
				+ cellAttributeArrayTest.length);

	}

	public static Mat toMat(float[] array) {

		// Mat returnMat = new Mat(130560, 1, CvType.CV_32FC1);
		Mat returnMat = new Mat(1, 1, CvType.CV_32FC1);

		for (int i = 0; i < array.length; i++) {
			Mat x = new Mat(1, 1, CvType.CV_32FC1);
			x.setTo(new Scalar(array[i]));
			returnMat.push_back(x);
			// returnMat.rowRange(i, i + 1).setTo(new Scalar(array[i]));
		}
		return returnMat;

	}

	public static float getAttributeClass(int x1, int y1, int x2, int y2,
			int[][] pointsArray, int[] attributeArray) {

		// // minX, maxX, minY, maxY
		float attributeValue = -1;
		int err = 16;

		for (int i = 0; i < pointsArray.length; i++) {

			if (x1 >= (pointsArray[i][0] - err)
					&& x1 <= (pointsArray[i][1] + err)) {

				if (x2 >= (pointsArray[i][0] - err)
						&& x2 <= (pointsArray[i][1] + err)) {

					if (y1 >= (pointsArray[i][2] - err)
							&& y1 <= (pointsArray[i][3] + err)) {

						if (y2 >= (pointsArray[i][2] - err)
								&& y2 <= (pointsArray[i][3] + err)) {

							attributeValue = attributeArray[i];
							return attributeValue;

						}

					}

				}

			}

		}

		// if the program ever comes here then there is some problem
		// with the association of the blocks
		// So we shall allocate the pixels rather than the blocks

		// lets allocate an array
		int[][] tempArray = new int[8][8];
		int tempFlag = 0;

		for (int row = y1, i = 0; row < y2; row++, i++) {

			for (int col = x1, j = 0; col < x2; col++, j++) {

				tempFlag = 0;

				// check which polygon does row,col belong to
				for (int k = 0; k < pointsArray.length; k++) {

					if (col >= pointsArray[k][0] && col <= pointsArray[k][1]) {

						if (row >= pointsArray[k][2]
								&& row <= pointsArray[k][3]) {

							// assign the temp array
							tempArray[i][j] = attributeArray[k];
							tempFlag = 1;
							break;

						}

					}

				}

				if (tempFlag == 1)
					continue;
				else
					tempArray[i][j] = -1;
			}

		}

		// now lets calculate the average
		int sum = 0;
		for (int row = 0; row < 8; row++) {
			for (int col = 0; col < 8; col++) {

				sum += tempArray[row][col];
			}
		}

		float avg = sum / (8 * 8);
		attributeValue = avg;
		return attributeValue;

	}

	public static MatOfFloat computeHog(Mat image) {

		MatOfFloat ders = new MatOfFloat();
		MatOfPoint locs = new MatOfPoint();

		// creating the hog descriptor for the sample
		HOGDescriptor hog = new HOGDescriptor(image.size(), new Size(64, 64),
				new Size(8, 8), new Size(8, 8), 9);

		// computing the descriptors for the sample
		hog.compute(image, ders, new Size(0, 0), new Size(0, 0), locs);

		return ders;

	}

	public static int prepareData(Mat ders, Mat data, int prevNumberOfCells,
			int totalNumberOfCells) {

		for (int i = 0; i < ders.rows(); i++) {

			data.push_back(ders.rowRange(i, i + 1));

		}

		return prevNumberOfCells + totalNumberOfCells;

	}

}
