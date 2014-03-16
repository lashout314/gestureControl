package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.Video;

import android.app.Activity;
import android.content.Context;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.TextView;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    HAND_RECT_COLOR     = new Scalar(255, 0, 0);
    private static final Scalar	   HAND_CONTOUR_COLOR  = new Scalar(0, 255, 0);
    private static final Scalar	   WHITE			   = new Scalar(255,255,255);
    
    private int					   mUndersamplingFactor = 4;
    private int					   mErosionKernelSize = 36;
    private float				   mMinBBArea = 0.01f;


    private Mat                    mRgba;
    private Mat                    mGray;
    
    private Mat					   mOutputImage;
    
    private Mat					   mRedPrev;
    private Mat					   mGreenPrev;
    private Mat					   mBluePrev;
    private Mat					   mRedDiff;
    private Mat					   mGreenDiff;
    private Mat					   mBlueDiff;
    
    private Mat					   mGrayDiff;
    private Mat					   mGrayPrev;
    
    private Mat					   mGrayMask;
    
    private Mat					   mGrayStatic;

    private Mat					   mGrayDiffMasked;

    private Mat					   mGrayInverseMask;

    private Mat					   mFreezeFrame;
    
    private Mat					   mMhi;
    private Mat					   mMmask;
    private Mat					   mMorientation;
    private Mat					   mSmask;
    private MatOfRect			   mSBoundingBox;
    
    
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    

    private int					   mFrameNum = 1;
    private int					   mLastFrame = 1;
    

    private float                  mRelativeFaceSizeMin   = 0.05f;
    private float                  mRelativeFaceSizeMax   = 0.5f;
    
    private int                    mAbsoluteFaceSizeMin   = 0;
    private int					   mAbsoluteFaceSizeMax	  = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;
    
    private MediaPlayer mediaPlayer;
    private TextView songName;
    private int mVolume = 6, maxVolume = 10, minVolume = 2;
    private double startTime = 0;
    private double endTime = 0;
    

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());


                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public FdActivity() {

    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCameraIndex(1);
        mOpenCvCameraView.setCvCameraViewListener(this);
        
        mediaPlayer = MediaPlayer.create(this, R.raw.levels);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
        mRedPrev = new Mat();
        mGreenPrev = new Mat();
        mBluePrev = new Mat();
        mRedDiff = new Mat();
        mGreenDiff = new Mat();
        mBlueDiff = new Mat();
        mGrayPrev = new Mat();
        mGrayDiff = new Mat();
        mGrayDiff = new Mat();
        mGrayDiffMasked = new Mat();
        mGrayStatic = new Mat();
        mGrayMask = new Mat();
        mGrayInverseMask = new Mat();
        mMhi = new Mat();
        mMmask = new Mat();
        mMorientation = new Mat();
        mSmask = new Mat();
        mOutputImage = new Mat();
        mFreezeFrame = new Mat();
        
        mSBoundingBox = new MatOfRect();
       
        
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        mRedPrev.release();
        mGreenPrev.release();
        mBluePrev.release();
        mRedDiff.release();
        mGreenDiff.release();
        mBlueDiff.release();
        mGrayPrev.release();
        mGrayDiff.release();
        mGrayDiff.release();

        mGrayDiffMasked.release();
        mGrayMask.release();
        mGrayInverseMask.release();
        mGrayStatic.release();
        mMhi.release();
        mMmask.release();
        mMorientation.release();
        mSmask.release();
        mSBoundingBox.release();
        mOutputImage.release();
        mFreezeFrame.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        
        Size origSize = mRgba.size();
        
        double shrinkFactor = 1/(double)mUndersamplingFactor;
        //Log.d(TAG,"shrinkFactor = "+shrinkFactor);
        
        Imgproc.resize(mRgba, mRgba, new Size(), shrinkFactor, shrinkFactor, Imgproc.INTER_LINEAR);
        Imgproc.resize(mGray, mGray, new Size(), shrinkFactor, shrinkFactor, Imgproc.INTER_LINEAR);
        
        int M,N;
        
        M = mGray.rows();
        N = mGray.cols();
        
        Mat mRed = new Mat();
        Mat mGreen = new Mat();
        Mat mBlue = new Mat();

        Core.extractChannel(mRgba, mRed, 0);
        Core.extractChannel(mRgba, mGreen, 1);
        Core.extractChannel(mRgba, mBlue, 2);
        
             
        
        if (mFrameNum == 1) {
        	mRedPrev = mRed.clone();
        	mGreenPrev = mGreen.clone();
        	mBluePrev = mBlue.clone();
        }
        
        Core.absdiff(mRed, mRedPrev, mRedDiff);
        Core.absdiff(mGreen, mGreenPrev, mGreenDiff);
        Core.absdiff(mBlue, mBluePrev, mBlueDiff);
        
        mRedPrev = mRed.clone();
        mGreenPrev = mGreen.clone();
        mBluePrev = mBlue.clone();
        

        
        Imgproc.threshold(mRedDiff, mRedDiff, 15, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(mGreenDiff, mGreenDiff, 15, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(mBlueDiff, mBlueDiff, 15, 255, Imgproc.THRESH_BINARY);
        
        List<Mat> mRgbaSplit = new ArrayList<Mat>(3);
        
        
        mRgbaSplit.add(0, mRedDiff);
        mRgbaSplit.add(1, mGreenDiff);
        mRgbaSplit.add(2, mBlueDiff);
        
        
        
        Core.merge(mRgbaSplit, mRedDiff);
        Imgproc.cvtColor(mRedDiff, mGrayDiff, Imgproc.COLOR_RGB2GRAY);
        

        

        
        if (mFrameNum==1) {
        	Log.d(TAG,"mFrameNum=" + mFrameNum);
        	mMhi = Mat.zeros(mGrayDiff.size(), CvType.CV_32F);
        	mFreezeFrame = mGrayDiff.clone();
        }
        
        int kernelsize = Math.round(mErosionKernelSize/mUndersamplingFactor);
        //Log.d(TAG,"kernelsize = "+kernelsize);
        
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(kernelsize,kernelsize));
        
        Imgproc.erode(mGrayDiff, mGrayMask, kernel);
        
        for (int i = 1; i<=3; i++) {
        	Imgproc.dilate(mGrayMask, mGrayMask, kernel);
        }
        
        Imgproc.threshold(mGrayMask, mGrayMask, 25, 255, Imgproc.THRESH_BINARY);
        
        
//        mGrayDiffMasked = Mat.zeros(mGrayDiff.size(), mGrayDiff.type());
//        mGrayDiff.copyTo(mGrayDiffMasked, mGrayMask);
        
//        Mat tmp = new Mat();
//
//        Core.scaleAdd(Mat.ones(mGrayDiff.size(), mGrayDiff.type()), 255, Mat.zeros(mGrayDiff.size(), mGrayDiff.type()), tmp);
//        Core.subtract(tmp, mGrayDiff, mGrayInverseMask);
//        
//
//
//        mGray.copyTo(mGrayStatic,mGrayInverseMask);
        
        
        //Log.d(TAG,"size of mGrayMask="+mGrayMask.size()+" size of mMhi="+mMhi.size());
		Video.updateMotionHistory(mGrayMask, mMhi, mFrameNum, 1);
		
		
		Video.calcMotionGradient(mMhi, mMmask, mMorientation, 1, 1, 3);
			
		Video.segmentMotion(mMhi, mSmask, mSBoundingBox, mFrameNum, 1);

//		int height = mGray.rows();
//        if (mAbsoluteFaceSizeMin == 0) {  
//            if (Math.round(height * mRelativeFaceSizeMin) > 0) {
//                mAbsoluteFaceSizeMin = Math.round(height * mRelativeFaceSizeMin);
//            }
//        }
//        
//        if (mAbsoluteFaceSizeMax == 0) {
//            if (Math.round(height * mRelativeFaceSizeMax) > 0) {
//                mAbsoluteFaceSizeMax = Math.round(height * mRelativeFaceSizeMax);
//            }
//        }
//
//        MatOfRect faces = new MatOfRect();
//        
//        if (mJavaDetector != null)
//        	mJavaDetector.detectMultiScale(mGrayStatic, faces, 1.1, 2, 2,
//        			new Size(mAbsoluteFaceSizeMin, mAbsoluteFaceSizeMin), new Size(mAbsoluteFaceSizeMax, mAbsoluteFaceSizeMax));
//     
//        
//        Rect[] facesArray = faces.toArray();
//        
//        for (int i = 0; i< facesArray.length; i++) {
//        	Core.rectangle(mGray, facesArray[i].tl(), facesArray[i].br(), HAND_RECT_COLOR, 3);
//        	Log.d(TAG, "Face Detected.");
//        }

        
        Rect[] motionArray = mSBoundingBox.toArray();
        double orientationAngle, orientationDegrees, orientationRadians;
        double amplitudeInOrientation;
        double whratio;

        
        Mat localOrientation, localMmask, localMhi;
        

        
        double minArea = mMinBBArea * mGrayDiff.rows()*mGrayDiff.cols();


        for (int i = 0; i < motionArray.length; i++) {
        	
        
        	
        	localOrientation = mMorientation.submat(motionArray[i]);
        	localMmask = mMmask.submat(motionArray[i]);
        	localMhi = mMhi.submat(motionArray[i]);
        			
        	orientationAngle = Video.calcGlobalOrientation(localOrientation, localMmask, localMhi, mFrameNum, 1);
        	orientationDegrees = orientationAngle;
        	
        	if (orientationAngle>=270)
        		orientationAngle = 360 - orientationAngle;
        	else if (orientationAngle>=180)
        		orientationAngle = orientationAngle - 180;
        	else if (orientationAngle>90) {
        		orientationAngle = 180 - orientationAngle;
        		//mediaPlayer.start();
        	}
        	
        	orientationRadians = orientationAngle * 2 * (Math.PI) / 360;
        	
        	if (orientationRadians < Math.atan2(motionArray[i].height,motionArray[i].width)) {
        		amplitudeInOrientation = (motionArray[i].width/Math.cos(orientationRadians));
        	}
        	else {
        		amplitudeInOrientation = (motionArray[i].height/Math.sin(orientationRadians));
        	}
        	
        	if (motionArray[i].width > motionArray[i].height)
        		whratio = motionArray[i].width/motionArray[i].height;
        	else
        		whratio = motionArray[i].height/motionArray[i].width;

        	double aionormallized = 1000*amplitudeInOrientation/(Math.sqrt(motionArray[i].area()));
        	
        	if (aionormallized>1000 && motionArray[i].area()>minArea) {
        		
                mFreezeFrame = mGrayDiff.clone();
                Imgproc.cvtColor(mFreezeFrame, mFreezeFrame, Imgproc.COLOR_GRAY2RGB);
        		
        		mLastFrame = mFrameNum;
        		
        		Log.d(TAG, "orientationAngle=" + Math.round(orientationDegrees) + " width=" + motionArray[i].width + " height=" + motionArray[i].height + " aO=" + aionormallized);
        		
        		int x,y,w,h;
        		
        		x = motionArray[i].x;
        		y = motionArray[i].y;
        		w = motionArray[i].width;
        		h = motionArray[i].height;
        		
        		Rect handBBRect,origHandBBRect;
        		double handBBRectMargin = 1;
        		
        		
        		if (orientationDegrees>270+45)
        			handBBRect = new Rect(x+w-h,y,(int)(handBBRectMargin*h),h);
        		else if (orientationDegrees>180+45)
        			handBBRect = new Rect(x,y,w,(int)(handBBRectMargin*w));
        		else if (orientationDegrees>90+45)
        			handBBRect = new Rect(x,y,(int)(handBBRectMargin*h),h);
        		else if (orientationDegrees>0+45)
        			handBBRect = new Rect(x,y+h-w,w,(int)(handBBRectMargin*w));
        		else
        			handBBRect = new Rect(x+w-h,y,(int)(handBBRectMargin*h),h);
        		
        		boolean handBBRectValid = true;
        		
        		// double check new BB isn't out of bounds
        		if (handBBRect.x < 0)
        			handBBRectValid = false;
        		if (handBBRect.x>N)
        			handBBRectValid = false;
        		if (handBBRect.y < 0)
        			handBBRectValid = false;
        		if (handBBRect.y > M)
        			handBBRectValid = false;
        		if (handBBRect.x+handBBRect.width > N)
        			handBBRectValid = false;
        		if (handBBRect.y+handBBRect.height > M)
        			handBBRectValid = false;
        		
        		if (handBBRectValid==false) {
        			Log.d(TAG,"handBBRect Invalid"+handBBRect.x+","+handBBRect.y+","+handBBRect.width+","+handBBRect.height);
        			Log.d(TAG,"handBBRect M="+M+" N="+N);
        		}
        		
        		if (handBBRectValid) {
        			
        			
        		
	        		Mat handBoundingBox = mGrayDiff.submat(handBBRect).clone();
	        		
	        		mFreezeFrame = handBoundingBox.clone();
	        		Imgproc.cvtColor(mFreezeFrame, mFreezeFrame, Imgproc.COLOR_GRAY2RGB);
	        		
	        		//Imgproc.GaussianBlur(handBoundingBox, handBoundingBox, new Size(3, 3), 0);
	        		
	        		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
	        		
	        		Imgproc.findContours(handBoundingBox, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
	        		
	        		double maxArea = -1;
	        		int maxAreaIdx = -1;
	        		MatOfPoint contour = new MatOfPoint();
	        		
	        		for (int idx = 0; idx < contours.size(); idx++) {
	        			contour = contours.get(idx);
	        			double contourarea = Imgproc.contourArea(contour);
	        			if (contourarea > maxArea) {
	        				maxArea = contourarea;
	        				maxAreaIdx = idx;
	
	        				Log.i(TAG, "max contour idx: " + idx);
	        			}
	        		}
	        		
	        		MatOfInt hull = new MatOfInt();
	        		MatOfInt4 defects = new MatOfInt4();
	
	        		if (maxAreaIdx > -1) // contour detected
	        		{
	        	
		        		MatOfPoint2f new_contour = new MatOfPoint2f();
		        		double epsilon = 5;// debug here
		        		 
		        		contours.get(maxAreaIdx).convertTo(new_contour, CvType.CV_32FC2);
		        		//Processing on mMOP2f1 which is in type MatOfPoint2f
		        		Imgproc.approxPolyDP(new_contour, new_contour, epsilon, true); 
		        		//Convert back to MatOfPoint and put the new values back into the contours list
		        		new_contour.convertTo(contour, CvType.CV_32S); 
		        		 
		        		Imgproc.convexHull(contour, hull);
		
		        		Imgproc.convexityDefects(contour, hull, defects);
		
		        		long defect_num = defects.total();
		        		
		        		// hull to matofpoint
		        		MatOfPoint mopOut = new MatOfPoint();
		        		mopOut.create((int)hull.size().height,1,CvType.CV_32SC2);
	
		        		for(int j = 0; j < hull.size().height ; j++)
		        		{
		        		   int index = (int)hull.get(j, 0)[0];
		        		   double[] point = new double[] {
		        		       contour.get(index, 0)[0], contour.get(index, 0)[1]
		        		   };
		        		   mopOut.put(j, 0, point);
		        		}        
		        		
		        		List <MatOfPoint> hull_list = new ArrayList<MatOfPoint>();
		        		
		        		hull_list.add(0,mopOut);
		        		
		        		Imgproc.drawContours(mFreezeFrame, hull_list, 0, HAND_CONTOUR_COLOR, 2);
		        		
		        		// defect point reduce
		        		List<Point> vertex_point_list = new ArrayList<Point>();
		        		List<Point> defect_point_list = new ArrayList<Point>();
		        		int[] depth_list = new int[(int) defect_num];

		        		int DEPTH_THRESHOLD = 30000;
		        		int real_defect_num = 0;

		        		double[] point = new double[] { 0, 0 };
		        		Point current = new Point(point);

		        		int depth, index;
		        		
		        		Log.d(TAG,"Defect_num = "+ defect_num);
		        		
		        		for (int j = 0; j < (int) defect_num; j++) {

			        		depth = (int) defects.get(j, 0)[3];
//			        		if (depth < DEPTH_THRESHOLD)
//			        			continue;
	
			        		Log.d(TAG, "defect depth=" + depth);
			        		depth_list[real_defect_num] = depth;
			        		real_defect_num++;
	
			        		index = (int) defects.get(j, 0)[0];
			        		 
			        		current = new Point(contour.get(index, 0)[0],contour.get(index, 0)[1]);
			        		 
			        		// current.x = contour.get(index, 0)[0];
			        		// current.y = contour.get(index, 0)[1];
	
			        		vertex_point_list.add(current);
	
			        		index = (int) defects.get(j, 0)[2];
	
			        		// current.x = contour.get(index, 0)[0];
			        		// current.y = contour.get(index, 0)[1];
			        		current = new Point(contour.get(index, 0)[0],contour.get(index, 0)[1]);
			        		defect_point_list.add(current);

		        		}
		        		 
		        		Point p = new Point();
		        		p.x = 100;
		        		p.y = 100;
		        		Core.putText(mFreezeFrame, Integer.toString(real_defect_num-1), p,
		        		Core.FONT_HERSHEY_SIMPLEX, 2, WHITE, 3);
		        		 
		        		System.out.printf("real number defect =%d\n",real_defect_num);

		        		for (int k = 0; k < real_defect_num; k++) {
		        		Point vertex = vertex_point_list.get(k);
		        		Point defect = defect_point_list.get(k);
		        		Core.circle(mFreezeFrame, vertex, 2, new Scalar(10, 25, 155), 2);
		        		Core.circle(mFreezeFrame, defect, 2, new Scalar(280, 0, 55), 2);

		        		// double fontScale = 2;
		        		// Core.putText(rgba, Integer.toString(depth_list[i]), defect,
		        		// Core.FONT_HERSHEY_SIMPLEX, fontScale, mWhilte, 3);

		        		}
	        		}
        		}
        		
        		Core.rectangle(mFreezeFrame, handBBRect.tl(), handBBRect.br(), HAND_RECT_COLOR, 2);
        		
                
        	}
        }
            

        
        
        Mat flippedOutput = new Mat();
        
		Core.flip(mFreezeFrame, flippedOutput, 1);

		//Imgproc.resize(flippedOutput, mOutputImage, new Size(), mUndersamplingFactor, mUndersamplingFactor, Imgproc.INTER_LINEAR);
		Imgproc.resize(flippedOutput, mOutputImage, origSize, 0, 0, Imgproc.INTER_LINEAR);
		
		
        mRed.release();
        mGreen.release();
        mBlue.release();
        flippedOutput.release();
//        tmp.release();

        kernel.release();
        
//        faces.release();
        mRgbaSplit.clear();
        
        mFrameNum ++;
		
        return mOutputImage;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {

        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
   
        return true;
    }

    
}
