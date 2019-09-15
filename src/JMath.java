import java.util.Random;
/* Class for performing functions normally done by NumPy. */
class JMath {
    JMath(){

    }

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
        double[] hStacked = new double[arr1.length + arr2.length];
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
            for (int j = 0; j < x[0].length; j++){
                product[i] += x[i][j] *y[j];
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
    /* Adds the indexes together and stores the results in a new
     * 1d array.
     */
    public static double[] add1dArray(double[] arr1, double[] arr2){
        double[] result = new double[arr1.length];
        for(int i = 0; i < arr1.length; i++){
            result[i] = arr1[i] + arr2[i];
        }
        return result;
    }
    /* Subtracts the indexes together and stores the results in a new
     * 2d array.
     */
    public static double[][] sub2dArray(double[][] arr1, double[][] arr2){
        double[][] result = new double[arr1.length][arr1[0].length];
        for(int i = 0; i < arr1.length; i++){
            for (int j = 0; j < arr1[0].length; j++){
                result[i][j] = arr1[i][j] - arr2[i][j];
            }
        }
        return result;
    }
    //Todo: finish this method
    public static double[][] atleast2d(double[] dotProduct) {
        double[][] result = new double[dotProduct.length][dotProduct.length];
        result[0] = dotProduct;
        return result;
    }
    public static double sum1D(double[] arr){
        double sum = 0;
        for (int i = 0 ; i < arr.length ; i++){
            sum += arr[i];
        }
        return sum;
    }
    /* Returns a new array with the extra 1st dimension size and all the original values. */
    public static double[][] array2dReSize(double[][] array, int newLength){
        if (array.length > newLength){
            return array;
        } else {
            double[][] newArray = new double[newLength][array[0].length];
            for (int i = 0; i < array.length; i++){
                newArray[i] = array[i];
            }
            return newArray;
        }
    }
    /* Divides every element in the array by a scalar */
    public static double[][] divide2d(double[][] arr, double scalar){
        for (int i = 0; i<arr.length;i++){
            for (int j = 0; j<arr[0].length;j++){
                arr[i][j] /= scalar;
            }
        }
        return arr;
    }
    /* M every element in the array by a scalar */
    public static double[][] multiply2d(double[][] arr, double scalar){
        for (int i = 0; i<arr.length;i++){
            for (int j = 0; j<arr[0].length;j++){
                arr[i][j] *= scalar;
            }
        }
        return arr;
    }
    /* adds number to each element in the array  */
    public static double[][] add2d(double[][] arr, double number){
        for (int i = 0; i<arr.length;i++){
            for (int j = 0; j<arr[0].length;j++){
                arr[i][j] += number;
            }
        }
        return arr;
    }
    /* Raises each element by a given power */
    public static double[][] power2d(double[][] arr, double power){
        for (int i = 0; i<arr.length;i++){
            for (int j = 0; j<arr[0].length;j++){
                arr[i][j] = Math.pow(arr[i][j], power);
            }
        }
        return arr;
    }
    /* num is divided by each element in the array */
    public static double[][] dDivide(double[][] arr, double num){
        for (int i = 0; i<arr.length;i++){
            for (int j = 0; j<arr[0].length;j++){
                arr[i][j] = Math.pow(arr[i][j], num);
            }
        }
        return arr;

    }
    /*Returns the indicie of the largest value */
    public static int argMax(double[] arr){
        int largest = 0;
        for (int i = 0; i < arr.length; i++){
            if (arr[largest]< arr[i]){
                largest = i;
            }
        }
        return largest;
    }
    /* Ensures all values are with in a certain range to prevent escaping gradient problem. */
    public static double[] clip(double[] arr, double upperLimit, double lowerLimit){
        for (int i = 0 ; i < arr.length; i++){
            if (arr[i] > upperLimit){
                arr[i] = upperLimit;
            } else if (arr[i]< lowerLimit){
                arr[i] = lowerLimit;
            }
        }
        return arr;
    }
    public static double[] subArray(double[] arr, int lower, int upper){
        double[] newArray = new double[arr.length-lower-(arr.length-upper)];
        for (int i = lower; i<upper; i++){
            newArray[i-lower] = arr[i];
        }
        return newArray;
    }
    public static double[][] divideByArray(double num, double[][] arr){
        double[][] result = new double[arr.length][arr[0].length];
        for (int i = 0; i < arr.length; i++){
            for (int j = 0; j < arr[0].length; j++) {
                result[i][j] = num / arr[i][j];
            }
        }
        return result;
    }
}
