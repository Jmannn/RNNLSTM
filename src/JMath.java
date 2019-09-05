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
    /* Concatenates two multi dimensional arrays, only concat first dimension.
     * It is basically stacking the rows. */
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
    /* Concatenates two multi dimensional arrays by adding to the end of rows to make larger rows */
    public static double[] hStack(double[] arr1, double[] arr2){
        double[] hStacked = new double[arr1.length];
        for (int i = 0; i < arr1.length; i++){
            hStacked[i] = arr1[i];
        }
        for (int i = arr1.length; i < arr2.length + arr1.length; i++){
            hStacked[i] = arr2[i-arr1.length];
        }
        return hStacked;
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
    /* For 1d by 1d */
    public static double[] dotProduct(double[] x, double[] y){

        double[] product = new double[x.length];

        for (int i = 0 ; i < x.length; i++){
            product[i] = y[i] * x[i];

        }
        return product;
    }
    /* For 2d by 2d arrays*/
    public static double[][] dotProduct(double[][] x, double[][] y){

        double[][] product = new double[x.length][x[0].length];

        for (int i = 0 ; i < x.length; i++){
            for (int j = 0; j < x[0].length; j++){
                product[i][j] = x[i][j] * y[i][j];
            }
        }
        return product;
    }
    /* Calculates the difference from one 1d array to another. */
    public static double[] difference(double[] arr1, double[] arr2){
        double[] difference = new double[arr1.length];

        for(int i = 0; i < arr1.length; i++){
            difference[i] = Math.abs(arr1[i] - arr2[i]);
        }
        return difference;
    }
    /* Adds the indexes together and stores the results in a new
     * 2d array.
     */
    public static double[][] add2dArray(double[][] arr1, double[][] arr2){
        double[][] result = new double[arr1.length][arr1[0].length];
        for(int i = 0; i < arr1.length; i++){
            for (int j = 0; j < arr1[0].length; j++){
                result[i][j] = arr1[i][j] + arr2[i][j];
            }
        }
        return result;
    }

    public static double[][] atleast2d(double[] dotProduct) {
        double[][] result = new double[dotProduct.length][dotProduct.length];
        result[0] = dotProduct;
        return result;
    }
}
