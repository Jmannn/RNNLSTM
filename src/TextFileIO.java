import java.io.File;
import java.util.Scanner;
import java.util.List;
import java.util.ArrayList;

public class TextFileIO {

    private File file;
    private Scanner scan;



    private char[] text;
    private char[] uniqueCharArray;
    private int[][] returnData;
    private int[][] output;
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
            this.scan = new Scanner(this.file);
            while (this.scan.hasNext()){
                word = scan.next();
                for (int i = 0; i < word.length(); i++){
                    characterList.add(word.charAt(i));
                    if (!uniqueCharList.contains(word.charAt(i))){
                        uniqueCharList.add(word.charAt(i));
                    }
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

            returnData = new int[uniqueWordsNum+1][dataSize];
            for (int i = 0; i < dataSize; i++){
                returnData[i][i] = 1;
            }
            output = new int[uniqueWordsNum][outputSize];
            int index =-1;
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < uniqueCharArray.length; j++) {
                    if (text[i] == uniqueCharArray[j]){
                        index = j;
                    }
                }
                output[i] = returnData[index];
            }
            this.returnData = returnData;
            this.output = output;
            this.outputSize = outputSize;
            this.data = uniqueCharArray;
            this.uniqueWordsNum = uniqueCharArray.length;

        } catch (Exception e){
            System.err.println("IO read failed");
            e.printStackTrace();
        }

    }
    public int[][] getReturnData() {
        return returnData;
    }
    public char[] getText() {
        return text;
    }
    public int[][] getOutput() {
        return output;
    }
    public int getUniqueWordsNum() {
        return uniqueWordsNum;
    }
    public int getOutputSize() {
        return outputSize;
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