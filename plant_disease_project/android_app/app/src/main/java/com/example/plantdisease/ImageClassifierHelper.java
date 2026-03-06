package com.example.plantdisease;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;

import java.io.IOException;
import java.util.List;

public class ImageClassifierHelper {
    private static final String TAG = "ImageClassifierHelper";
    
    // Model parameters
    private static final String MODEL_NAME = "model_quantized.tflite"; // Load quantized model to match performance test
    private static final int MAX_RESULTS = 1;
    private static final float THRESHOLD = 0.1f;
    private static final int THREADS = 2;

    private final Context context;
    private final ClassifierListener classifierListener;
    private ImageClassifier imageClassifier;

    public interface ClassifierListener {
        void onError(String error);
        void onResults(List<Classifications> results, long inferenceTime);
    }

    public ImageClassifierHelper(Context context, ClassifierListener listener) {
        this.context = context;
        this.classifierListener = listener;
        setupImageClassifier();
    }

    private void setupImageClassifier() {
        ImageClassifier.ImageClassifierOptions.Builder optionsBuilder =
                ImageClassifier.ImageClassifierOptions.builder()
                        .setScoreThreshold(THRESHOLD)
                        .setMaxResults(MAX_RESULTS);

        BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder().setNumThreads(THREADS);
        
        // Optional: Use GPU delegate if supported
        if (new CompatibilityList().isDelegateSupportedOnThisDevice()) {
             baseOptionsBuilder.useGpu();
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build());

        try {
            imageClassifier = ImageClassifier.createFromFileAndOptions(
                    context, MODEL_NAME, optionsBuilder.build());
        } catch (IOException e) {
            String error = "TFLite failed to load model with error: " + e.getMessage();
            Log.e(TAG, error);
            classifierListener.onError(error);
        }
    }

    public void classify(Bitmap bitmap, int imageRotation) {
        if (imageClassifier == null) {
            setupImageClassifier();
        }

        // Image preprocessing
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                // The TFLite task library handles resizing and normalizations defined in TFLite metadata
                // But we can add a resize op here if needed - metadata is usually missing when converting raw keras
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .build();

        TensorImage tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap));

        long startTime = SystemClock.uptimeMillis();
        // Inference
        List<Classifications> results = imageClassifier.classify(tensorImage);
        long inferenceTime = SystemClock.uptimeMillis() - startTime;

        classifierListener.onResults(results, inferenceTime);
    }

    public void clearImageClassifier() {
        if (imageClassifier != null) {
            imageClassifier.close();
            imageClassifier = null;
        }
    }
}
