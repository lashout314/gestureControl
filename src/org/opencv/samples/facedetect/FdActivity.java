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
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.Video;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(255, 255, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    private Mat                    mRgba;
    private Mat					   mRedPrev;
    private Mat					   mGreenPrev;
    private Mat					   mBluePrev;
    private Mat					   mRedDiff;
    private Mat					   mGreenDiff;
    private Mat					   mBlueDiff;
    private Mat                    mGray;
    private Mat					   mGrayFull;
    private Mat					   mGrayPrev;
    private Mat					   mGrayDiff;
    private Mat					   mGrayDiffMasked;
    private Mat					   mGrayMask;
    private Mat					   mMhi;
    private Mat					   mMmask;
    private Mat					   mMorientation;
    private Mat					   mSmask;
    private MatOfRect			   mSBoundingBox;
    
    
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;
    

    private int                    mDetectorType       = JAVA_DETECTOR;
    private int					   mFrameNum = 1;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;

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
                        InputStream is = getResources().openRawResource(R.raw.cascade);
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

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

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
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
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
        mGrayFull = new Mat();
        mGrayPrev = new Mat();
        mGrayDiff = new Mat();
        mGrayDiffMasked = new Mat();
        mGrayMask = new Mat();
        mMhi = new Mat();
        mMmask = new Mat();
        mMorientation = new Mat();
        mSmask = new Mat();
        
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
        mGrayFull.release();
        mGrayDiffMasked.release();
        mGrayMask.release();
        mMhi.release();
        mMmask.release();
        mMorientation.release();
        mSmask.release();
        mSBoundingBox.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        
        Imgproc.resize(mRgba, mRgba, new Size(), 0.25, 0.25, Imgproc.INTER_LINEAR);
        Imgproc.resize(mGray, mGray, new Size(), 0.25, 0.25, Imgproc.INTER_LINEAR);
        
        List<Mat> mRgbaSplit = new ArrayList<Mat>(3);

        
        Core.split(mRgba, mRgbaSplit);
              
        Mat	mRed = mRgbaSplit.get(0);
        Mat mGreen = mRgbaSplit.get(1);
        Mat mBlue = mRgbaSplit.get(2);
        
        if (mFrameNum == 1) {
        	mRedPrev = mRed.clone();
        	mGreenPrev = mGreen.clone();
        	mBluePrev = mBlue.clone();
        	mMhi = Mat.zeros(mGray.size(), CvType.CV_32F);
        }
        
        Core.absdiff(mRed, mRedPrev, mRedDiff);
        Core.absdiff(mGreen, mGreenPrev, mGreenDiff);
        Core.absdiff(mBlue, mBluePrev, mBlueDiff);
        
        mRedPrev = mRed.clone();
        mGreenPrev = mGreen.clone();
        mBluePrev = mBlue.clone();
        
        mFrameNum ++;
        
        Imgproc.threshold(mRedDiff, mRedDiff, 15, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(mGreenDiff, mGreenDiff, 15, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(mBlueDiff, mBlueDiff, 15, 255, Imgproc.THRESH_BINARY);
        
        mRgbaSplit.set(0, mRedDiff);
        mRgbaSplit.set(1, mGreenDiff);
        mRgbaSplit.set(2, mBlueDiff);
        
        Core.merge(mRgbaSplit, mRedDiff);
        Imgproc.cvtColor(mRedDiff, mGrayDiff, Imgproc.COLOR_RGB2GRAY);
        
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(23,23));
        
        Imgproc.erode(mGrayDiff, mGrayMask, kernel);
        
        for (int i = 1; i<=3; i++) {
        	Imgproc.dilate(mGrayMask, mGrayMask, kernel);
        }
        
        mGrayDiff.copyTo(mGrayDiffMasked, mGrayMask);
        
		Video.updateMotionHistory(mGrayDiffMasked, mMhi, mFrameNum, 1);
		
		
		Video.calcMotionGradient(mMhi, mMmask, mMorientation, 1, 1, 3);
			
		Video.segmentMotion(mMhi, mSmask, mSBoundingBox, mFrameNum, 1);


//        if (mAbsoluteFaceSize == 0) {
//            int height = mGray.rows();
//            if (Math.round(height * mRelativeFaceSize) > 0) {
//                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
//            }
//            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
//        }
////
//        MatOfRect faces = new MatOfRect();
//
//        if (mDetectorType == JAVA_DETECTOR) {
//            if (mJavaDetector != null)
//                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
//                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
//        }
//        else if (mDetectorType == NATIVE_DETECTOR) {
//            if (mNativeDetector != null)
//                mNativeDetector.detect(mGray, faces);
//        }
//        else {
//            Log.e(TAG, "Detection method is not selected!");
//        }
//
//        
//        Rect[] facesArray = faces.toArray();
//        
//        for (int i = 0; i< facesArray.length; i++) {
//        	Core.rectangle(mGray, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
//        	Log.d(TAG, "Face Detected.");
//        }
//                
        
        Rect[] motionArray = mSBoundingBox.toArray();
        double orientationAngle, orientationDegrees, orientationRadians;
        double amplitudeInOrientation;
        double whratio;
        
        
        Mat localOrientation, localMmask, localMhi;
        
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
        	else if (orientationAngle>90)
        		orientationAngle = 180 - orientationAngle;
        	
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

        	
        	
        	
        	if (amplitudeInOrientation>70) {
        		Core.rectangle(mGrayDiffMasked, motionArray[i].tl(), motionArray[i].br(), FACE_RECT_COLOR, 3);
        		Log.d(TAG, "orientationAngle=" + Math.round(orientationDegrees) + " width=" + motionArray[i].width + " height=" + motionArray[i].height + " aO=" + amplitudeInOrientation);
        		
        	}
        }
            

		Core.flip(mGrayDiffMasked, mGrayDiffMasked, 1);
		
		Imgproc.resize(mGrayDiffMasked, mGrayDiffMasked, new Size(), 4, 4, Imgproc.INTER_LINEAR);
		
//		Core.flip(mGrayDiffMasked, mGrayDiffMasked, 1);
//		
//		Imgproc.resize(mGrayDiffMasked, mGrayDiffMasked, new Size(), 4, 4, Imgproc.INTER_LINEAR);
		
		
        return mGrayDiffMasked;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }
}
