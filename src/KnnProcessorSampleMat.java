import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.ml.CvKNearest;


//java version for
//http://blog.damiles.com/2008/11/the-basic-patter-recognition-and-classification-with-opencv/

public class KnnProcessorSampleMat {
	
	public static void main(String [] args){
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		int trainingSamples = 300;
		Mat trainData =  new Mat(trainingSamples,2,CvType.CV_32FC1);
		Mat trainClasses = new Mat(trainingSamples, 1 , CvType.CV_32FC1);
		
		trainClasses.rowRange(0, 100).setTo( new Scalar(1.0));
		trainClasses.rowRange(101, 200).setTo( new Scalar(2.0));
		trainClasses.rowRange(201, 300).setTo( new Scalar(3.0));
		//System.out.println(trainClasses.dump());
		trainData.setTo(new Scalar(0.0));
		Core.randn(trainData.rowRange(0, 100), 200.0 , 50.0);
		System.out.println(trainData.dump());
		Core.randn(trainData.rowRange(101, 200), 400.0 , 50.0);
		Core.randn(trainData.rowRange(201, 300), 600.0 , 50.0);
		
		//System.out.println(trainData.dump());
		
		//creating an object of the CvKnearest class
		CvKNearest knn = new CvKNearest(trainData, trainClasses);
		
		Mat testData = new Mat(500, 2 , CvType.CV_32FC1);
		Core.randn(testData, 300, 200);
		Mat results = new Mat(500, 2 , CvType.CV_32FC1);
		Mat neighborResponses = new Mat(500, 2 , CvType.CV_32FC1);
		Mat dists = new Mat(500, 2 , CvType.CV_32FC1);
		
		knn.find_nearest(testData, 5 , results, neighborResponses, dists);
		
		System.out.println("Test Data----------------------------------------");
		System.out.println(testData.dump());
		ImageShow im = new ImageShow("TestData");
		im.showImage(testData);
		Highgui.imwrite("testData.png", testData);
		System.out.println("Results ----------------------------------------");
		System.out.println(results.size().toString());
		System.out.println(" Neighbor Responses ----------------------------------------");
		System.out.println(neighborResponses.dump());
		Highgui.imwrite("neighbor.png", neighborResponses);
		System.out.println("Dists ----------------------------------------");
		System.out.println(dists.dump());
		
			}

}