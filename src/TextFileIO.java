
import java.io.*;
import java.nio.charset.Charset;
import java.util.Scanner;
import java.util.List;
import java.util.ArrayList;
import java.lang.StringBuilder;

public class TextFileIO {

    private File file;
    private Scanner scan;



    private char[] text;
    private char[] uniqueCharArray;
    private double[][] returnData;
    private double[][] output;
    private int outputSize;
    private int uniqueWordsNum;
    private int dataSize;
    private char[] data;

    /* Constructor which takes in a filename,
     * and extracts processed data from the file. */
    public TextFileIO(String fileName){
        try {
            List<Character> characterList = new ArrayList<>();
            List<Character> uniqueCharList = new ArrayList<>();
            String word = "";

            this.file = new File(fileName);
            Charset encoding = Charset.defaultCharset();
            InputStream in = new FileInputStream(this.file);
            Reader reader = new InputStreamReader(in, encoding);
            // buffer for efficiency
            Reader buffer = new BufferedReader(reader);
            char c = ' ';
            int r;
            while ((r =  buffer.read()) != -1){
                c = (char) r;
                characterList.add(c);
                System.out.println(c);
                if (!uniqueCharList.contains(c)){
                    uniqueCharList.add(c);
                }
            }
            this.text = new char[characterList.size()];
            for (int i = 0; i < characterList.size(); i++) {
                this.text[i] = characterList.get(i).charValue();
            }
            this.uniqueCharArray = new char[uniqueCharList.size()];
            for (int i = 0; i < uniqueCharList.size(); i++) {
                this.uniqueCharArray[i] = uniqueCharList.get(i).charValue();
            }

            this.outputSize = text.length;
            this.uniqueWordsNum = dataSize = uniqueCharArray.length;

            returnData = new double[uniqueWordsNum+1][dataSize];
            for (int i = 0; i < dataSize; i++){
                returnData[i][i] = 1;
            }
            output = new double[uniqueWordsNum][outputSize];
            int index =-1;
            //loop should be output size
            //System.out.println("output: S: "+outputSize);
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < uniqueCharArray.length; j++) {
                    if (text[i] == uniqueCharArray[j]){
                        index = j;
                    }
                }
                //System.out.println(output.length + " " + output[0].length);
                //this does not work because in python you are extracting a row
                //and store it in row
                //also, it is the ravel of the column
                output = JMath.addColumn(returnData, output, index, i);
            }
            this.returnData = returnData;
            this.output = output;
            this.outputSize = outputSize;
            System.out.println("outputSIz "+outputSize);
            this.data = uniqueCharArray;
            this.uniqueWordsNum = uniqueCharArray.length;

        } catch (Exception e){
            System.err.println("IO read failed");
            e.printStackTrace();
        }

    }
    /*

    for i in range(0, output.shape[0]):W
        for j in range(0, output.shape[1]):
            prob[j] = output[i][j] / np.sum(output[i])
        outputText += np.random.choice(data, p=prob)
    return

     */
    public void export(double[][] output, char[] data, String filename){
        StringBuilder outputText = new StringBuilder();//use .append() for char
        WeightedRandom wr;
        //JMath.printArray(output);
        JMath.containsNAN(output);
        JMath.containsInfinity(output);
        double[] prob = new double[output[0].length];


        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[i].length; j++) {
                prob[j] = output[i][j] / JMath.sum1D(output[i]);
            }

            //JMath.containsNAN(prob);
            wr = new WeightedRandom(data, prob);
            //outputText += np.random.choice(data, p=prob)
            outputText.append(wr.pick());

            //p assigns a choice to each one of the array indexes
        }
        System.out.println(outputText.toString());

        try  {
            PrintWriter pw = new PrintWriter(filename);
            pw.append("   Done!");
            pw.println(outputText.toString());
        } catch (FileNotFoundException e){
            e.printStackTrace();
        }
    }
    public double[][] getReturnData() {
        return returnData;
    }
    public char[] getText() {
        return text;
    }
    public double[][] getOutput() {
        return output;
    }
    public int getUniqueWordsNum() {
        return uniqueWordsNum;
    }
    public int getOutputSize() {
        return outputSize;
    }
    public char[] getData() {
        return data;
    }
}
/*
returnData = np.append(returnData, np.atleast_2d(data), axis=0)
////adds the charlist to the very end of the return data 2d array
    output = np.zeros((uniqueWords, outputSize))
    for i in range(0, outputSize):
        index = np.where(np.asarray(data) == text[i])
        output[:,i] = returnData[0:-1,index[0]].astype(float).ravel()
        ////ravel(): [[1 2 3],[2 3 4],[3 4 5]] becomes [1 2 3 2 3 4 3 4 5]
        //////makes a 2d array 1d by laying on the x axis
        ////astype(): is merely a cast as floating point number
        ////[lower:higher] is
        /////where: returns only the elements that satisfy a condition
        //////returns only the indices that satisfy the conditions
 */