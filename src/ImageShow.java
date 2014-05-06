import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class ImageShow {

	public JFrame Window;
	private ImageIcon image;
	private JLabel label;
	private MatOfByte matOfByte;
	private Boolean SizeCustom;
	private int Height, Width;

	public ImageShow(String title) {
		Window = new JFrame();
		image = new ImageIcon();
		label = new JLabel();
		matOfByte = new MatOfByte();
		label.setIcon(image);
		Window.getContentPane().add(label);
		Window.setResizable(false);
		Window.setTitle(title);
		SizeCustom = false;
	}

	public ImageShow(String title, int height, int width) {
		SizeCustom = true;
		Height = height;
		Width = width;

		Window = new JFrame();
		image = new ImageIcon();
		label = new JLabel();
		matOfByte = new MatOfByte();
		label.setIcon(image);
		Window.getContentPane().add(label);
		Window.setResizable(false);
		Window.setTitle(title);

	}

	public void showImage(Mat img) {
		if (SizeCustom) {
			Imgproc.resize(img, img, new Size(Height, Width));
		}
		Highgui.imencode(".png", img, matOfByte);
		byte[] byteArray = matOfByte.toArray();
		BufferedImage bufImage = null;
		try {
			InputStream in = new ByteArrayInputStream(byteArray);
			bufImage = ImageIO.read(in);
			image.setImage(bufImage);
			Window.pack();
			label.updateUI();
			Window.setVisible(true);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}