import java.util.Random;
/* Class for performing functions normally done by NumPy. */
public class JMath {
    JMath(){

    };

    /* Used to transpose a java array */
    public static double[][] transpose(double[][] array){
        double[][] arrayT = new double[array[0].length][array.length];
        for (int i = 0; i < array.length; i++){
            for (int j = 0; j < array[0].length; j++){
                arrayT[j][i] = array[i][j];
            }
        }
        return arrayT;
    }
    /* Initializes a 2d array of doubles with random values */
    public static double[][] randomSample2D(int x, int y){
        Random random = new Random();
        double[][] newArray = new double[x][y];
        for (int i = 0; i < x; i++){
            for (int j = 0; j < y; y++){
                newArray[i][j] = random.nextGaussian();
            }
        }
        return newArray;
    }
    /* Concatenates two multi dimensional arrays */
    public static double[][] vStack(double[][] arr1, double[][] arr2){
        double[][] vStacked = new double[arr1.length+arr2.length][arr1[0].length];
        for(int i = 0 ; i < arr1.length; i++){
            vStacked[i] = arr1[i];
        }
        for(int i = arr1.length; i<arr1.length+arr2.length; i++){
            vStacked[i] = arr2[i-arr1.length];
        }
        return vStacked;
    }
    /* For 1d array scalers dot 2d */
    public static double[] dotProduct(double[][] x, double[] y){

        double[] product = new double[x.length];

        for (int i = 0 ; i < x.length; i++){
            product[i] = y[i];
            for (int j = 0; j < x[0].length; j++){
                product[i] *= x[i][j];
            }
        }
        return product;
    }

}
