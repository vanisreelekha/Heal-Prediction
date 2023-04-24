package com.example.testmodel

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import com.example.testmodel.ml.MyModelMetadata
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import kotlin.math.max

class MainActivity : AppCompatActivity()
{
    lateinit var predictBtn : Button
    lateinit var resView: TextView
    lateinit var textInput: EditText

    val MODEL_PATH = "my_model_metadata.tflite";

    override fun onCreate(savedInstanceState: Bundle?)
    {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        predictBtn = findViewById(R.id.predict_button);
        textInput = findViewById(R.id.message_text);
        resView = findViewById(R.id.result_text);

        var labels = application.assets.open("label.txt").bufferedReader().readLines()

        predictBtn.setOnClickListener {

            val model = MyModelMetadata.newInstance(this)

            val myString = textInput.text.toString();
            val encodedBytes = myString.toByteArray(Charsets.UTF_8)
            val byteBuffer = ByteBuffer.allocate(1200);
            byteBuffer.put(encodedBytes);



// Creates inputs for reference.
            val inputText = TensorBuffer.createFixedSize(intArrayOf(1, 300), DataType.FLOAT32)
            inputText.loadBuffer(byteBuffer)

            print(inputText.toString())
// Runs model inference and gets result.
            val outputs = model.process(inputText)
            val probability = outputs.probabilityAsTensorBuffer.floatArray

            var maxIdx = 0
            probability.forEachIndexed { index, fl ->
                if (probability[maxIdx]< fl)
                    maxIdx = index
            }


            resView.text = labels[maxIdx];

// Releases model resources if no longer used.
            model.close()
        }



        /*//val option = NLClassifierOptions.builder().build();
        val nlClassifier = NLClassifier.createFromFile(this,MODEL_PATH);
        *//*val executor = ScheduledThreadPoolExecutor(1);

        executor.execute{
            val results = nlClassifier.classify("Im very sad");
            Log.d("Sadness",results[0].toString());
        }*/


    }

}