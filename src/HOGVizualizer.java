import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class HOGVizualizer {

	public Mat getHogDescriptorVisualImage(Mat originalImage, MatOfFloat ders,
			Size winSize, Size cellSize
	// scale factor of 1
	// viz factor of 10
	) {

		float[] descriptorValues = ders.toArray();

		Mat descriptorSet = new Mat();

		int gradientBinSize = 9;
		int cellsInX = (int) (winSize.width / cellSize.width);
		int cellsInY = (int) (winSize.height / cellSize.height);
		int[][] cellUpdatCounter = new int[cellsInY][cellsInX];
		float[][][] gradientStrengths = new float[cellsInY][cellsInX][gradientBinSize];

		
		int blocksinX = cellsInX - 1;
		int blocksinY = cellsInY - 1;

		// for getting the descriptor values per cell
		System.out.println(descriptorValues.length);
		int descriptorIdx = 0;
		int cellx = 0;
		int celly = 0;

		for (int blockx = 0; blockx < blocksinX; blockx++) {

			for (int blocky = 0; blocky < blocksinY; blocky++) {

				for (int cellNr = 0; cellNr < 4; cellNr++) {

					cellx = blockx;
					celly = blocky;
					if (cellNr == 1)
						celly++;
					if (cellNr == 2)
						cellx++;
					if (cellNr == 3) {
						cellx++;
						celly++;
					}

					for (int bin = 0; bin < gradientBinSize; bin++) {
						/* if (descriptorIdx < descriptorValues.length) { */
						float gradientStrength = descriptorValues[descriptorIdx];
						descriptorIdx++;
						gradientStrengths[celly][cellx][bin] += gradientStrength;
						/* } */

					}

					cellUpdatCounter[celly][cellx]++;

				}

			}

		}

		// computing the average strengths
		for (celly = 0; celly < cellsInY; celly++) {
			for (cellx = 0; cellx < cellsInX; cellx++) {

				float NrUpdatesForThisCell = (float) cellUpdatCounter[celly][cellx];

				// compute average gradient strenghts for each gradient bin
				// direction
				for (int bin = 0; bin < gradientBinSize; bin++) {
					gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
				}
			}

		}

		descriptorSet = this.toMat(gradientStrengths, cellsInX, cellsInY);

		// System.out.println(descriptorSet.dump());

		return descriptorSet;

	}

	public Mat toMat(float[][][] array, int cellsinX, int cellsinY) {

		int binSize = 9;
		int totalNumberOfCells = cellsinX * cellsinY;
		int numberOfCells = 0;
		Mat generatedMat = new Mat(totalNumberOfCells, 9, CvType.CV_32FC1);
		generatedMat.setTo(new Scalar(0.0));
		for (int i = 0; i < cellsinY; i++) {

			for (int j = 0; j < cellsinX; j++) {

				for (int k = 0; k < binSize; k++) {

					if (numberOfCells < totalNumberOfCells) {

						generatedMat.rowRange(numberOfCells, numberOfCells + 1)
								.colRange(k, k + 1)
								.setTo(new Scalar(array[i][j][k]));
					}

				}

				numberOfCells++;

			}

		}

		return generatedMat;

	}

}
