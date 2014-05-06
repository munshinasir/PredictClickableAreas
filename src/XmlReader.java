import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.w3c.dom.Node;
import org.w3c.dom.Element;
import java.io.File;

public class XmlReader {

	private int[][] pointsArray;
	private int[] attributeArray;

	public XmlReader() {

		attributeArray = null;
		pointsArray = null;
	}

	public void readAnnotations(File file) {

		/*
		 * File folderName = new File("data");
		 * System.out.println(folderName.listFiles
		 * ()[1].listFiles()[0].getName());
		 */
		DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
		DocumentBuilder dBuilder = null;

		// creating an array to store the points

		try {
			dBuilder = dbFactory.newDocumentBuilder();
			Document doc = dBuilder.parse(file);

			// optional, but recommended
			doc.getDocumentElement().normalize();

			NodeList objectList = doc.getElementsByTagName("object");

			// creating an array to hold all the values
			int[][][] polygonArray = new int[objectList.getLength()][4][2];
			this.attributeArray = new int[objectList.getLength()];

			// heirarchy is document-> object -> polygon -> points

			// assuming annotations have one attribute per polygon
			// assuming annotations have one polygon per object

			for (int i = 0; i < objectList.getLength(); i++) {

				Element object = (Element) objectList.item(i);
				if (object.getNodeType() == Node.ELEMENT_NODE) {

					NodeList polygonList = object
							.getElementsByTagName("polygon");

					// casting the polygon to the element type
					Element polygon = (Element) polygonList.item(0);

					Integer attribute = Integer.parseInt(object
							.getElementsByTagName("attributes").item(0)
							.getTextContent());

					this.attributeArray[i] = attribute;

					for (int j = 0; j < 4; j++) {

						Element pt = (Element) polygon.getElementsByTagName(
								"pt").item(j);

						Integer x = Integer.parseInt(pt
								.getElementsByTagName("x").item(0)
								.getTextContent());
						Integer y = Integer.parseInt(pt
								.getElementsByTagName("y").item(0)
								.getTextContent());

						polygonArray[i][j][0] = x;
						polygonArray[i][j][1] = y;

					}

				}
			}

			this.pointsArray = getArray(polygonArray, objectList.getLength());

		} catch (Exception e) {

			e.printStackTrace();
		}

	}

	public int[][] getPointsArray() {
		return pointsArray;
	}

	public void setPointsArray(int[][] pointsArray) {
		this.pointsArray = pointsArray;
	}

	public int[] getAttributeArray() {
		return attributeArray;
	}

	public void setAttributeArray(int[] attributeArray) {
		this.attributeArray = attributeArray;
	}

	// transform the the polygon and point to a descriptor of type
	// minX, maxX, minY, maxY
	public static int[][] getArray(int[][][] array, int len) {

		int[][] result = new int[len][4];

		for (int polygon = 0; polygon < len; polygon++) {

			int maxY = 0, maxX = 0;
			int minY = 2000, minX = 2000;

			int length = array[polygon].length;

			// System.out.println(length);

			for (int i = 0; i < length; i++) {

				// getting the boundary maximums
				if (array[polygon][i][0] > maxX)
					maxX = array[polygon][i][0];

				if (array[polygon][i][0] < minX)
					minX = array[polygon][i][0];

				if (array[polygon][i][1] > maxY)
					maxY = array[polygon][i][1];

				if (array[polygon][i][1] < minY)
					minY = array[polygon][i][1];

			}

			// adding that to the matrix
			result[polygon][0] = minX;
			result[polygon][1] = maxX;
			result[polygon][2] = minY;
			result[polygon][3] = maxY;
		}

		return result;

	}

}
