import java.io.*;
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
    // BROKEN !!
    public static double[][] randomSample2D(int x, int y){
        Random random = new Random();
        double[][] newArray = new double[x][y];
        for (int i = 0; i < x; i++){
            for (int j = 0; j < y; j++){
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
            vStacked[i] = copyArray(arr1[i]);
        }
        for(int i = arr1.length; i<arr1.length+arr2.length; i++){
            vStacked[i] = copyArray(arr2[i-arr1.length]);
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
            for (int j = 0; j < y.length; j++){
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
    public static double[][] dotProduct(double[][] x, double[][] y) {

        double[][] product = new double[x.length][x[0].length];
        //product = new [1][1]
        if (y.length == x[0].length && y[0].length == x.length && y[0].length == 1 && x.length == 1) {
            product = new double[1][1];
            for (int i = 0; i < x[0].length; i++) {
                product[0][0] += y[i][0] * x[0][i];
            }
        }else if((y.length == x[0].length && y[0].length != x.length && y.length == 1)){
            //Todo: build this
            product = new double[x.length][y[0].length];

            for (int i = 0; i < product.length; i++) {
                product[i] = copyArray(y[0]);
                for (int j = 0; j < product[0].length; j++) {
                    product[i][j] *= x[i][0];
                }
            }
        }else {
            for (int i = 0 ; i < x.length; i++){
                for (int j = 0; j < x[0].length; j++){
                    product[i][j] = x[i][j] * y[i][j];
                }
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

    //Todo: finish this method c
    public static double[][] atleast2d(double[] dotProduct) {
        double[][] result = new double[1][dotProduct.length];
        result[0] = copyArray(dotProduct);
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
    public static double[][] array2dReSize(double[][] arrayOR, int newLength){
        double[][] array = copyArray(arrayOR);
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
    public static double[][] divide2d(double[][] arrOR, double scalar){
        double[][] arr = copyArray(arrOR);
        for (int i = 0; i<arr.length;i++){
            for (int j = 0; j<arr[0].length;j++){
                arr[i][j] /= scalar;
            }
        }
        return arr;
    }
    /* M every element in the array by a scalar */
    public static double[][] multiply2d(double[][] arrOR, double scalar){
        double[][] arr = copyArray(arrOR);
        for (int i = 0; i<arr.length;i++){
            for (int j = 0; j<arr[0].length;j++){
                arr[i][j] *= scalar;
            }
        }
        return arr;
    }
    /* adds number to each element in the array  */
    public static double[][] add2d(double[][] arrOR, double number){
        double[][] arr = copyArray(arrOR);
        for (int i = 0; i<arr.length;i++){
            for (int j = 0; j<arr[0].length;j++){
                arr[i][j] += number;
            }
        }
        return arr;
    }
    /* Raises each element by a given power */
    public static double[][] power2d(double[][] arrOR, double power){
        double[][] arr = copyArray(arrOR);
        for (int i = 0; i<arr.length;i++){
            for (int j = 0; j<arr[0].length;j++){
                arr[i][j] = Math.pow(arr[i][j], power);
            }
        }
        return arr;
    }
    /* num is divided by each element in the array */
    public static double[][] dDivide(double[][] arrOR, double num){
        double[][] arr = copyArray(arrOR);
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
    public static double[] clip(double[] arrOr, double upperLimit, double lowerLimit){
        double[] arr = copyArray(arrOr);
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
    /* Takes double array as input and transfers the contents to a single horizontal array */
    public static double[] ravel(double[][] arr2D){
        double[] arr1D = new double[(arr2D.length * arr2D[0].length)];
        int firstDimensionPointer = 0;
        for (int i = 0; i < arr2D.length; i++){
            for (int j = 0; j < arr2D[0].length; j++) {
                arr1D[firstDimensionPointer] = arr2D[i][j];
                firstDimensionPointer++;
            }
        }
        return arr1D;
    }
    public static double[][] intToDouble2D(int[][] arr){
        double[][] result = new double[arr.length][arr[0].length];
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                result[i][j] = arr[i][j];
            }
        }
        return result;
    }
    // % modulus
    // sets col of arr2 to col of ar1
    //Todo: fix this, needs a clone method
    public static double[][] addColumn(double[][] reTurn, double[][] input, int index, int col2){
        //printArray(reTurn);
        //printArray(output);
        double[][] output = copyArray(input);
        for (int i = 0; i < output.length; i++) {
            output[i][col2] = reTurn[i][index];
        }
        return output;
    }
    public static void printArray(double[][] arr){
        System.out.println("********");
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                System.out.print(arr[i][j] + " ");
            }
            System.out.println("\n --------");

        }
        System.out.println("********");

    }
    public static void printArray(double[] arr){
        System.out.println("======================");
        for (int j = 0; j < arr.length; j++) {
            System.out.print(arr[j] + " ");
        }
        System.out.println("\n======================");
    }
    public static double[] copyArray(double[] arr){
        double[] clone = new double[arr.length];
        for (int i = 0; i < clone.length; i++) {
            clone[i] = arr[i];
        }
        return clone;
    }
    public static double[][] copyArray(double[][] arr){
        double[][] clone = new double[arr.length][arr[0].length];
        for (int i = 0; i < clone.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                clone[i][j] = arr[i][j];
            }
        }
        return clone;
    }
    public static void containsNAN(double[][] arr){
        boolean nan = false;
        int totalNAN = 0, total = 0;
        for (int i = 0; i < arr.length ; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                if (Double.isNaN(arr[i][j])){
                    nan = true;
                    totalNAN++;
                    printArray(arr);
                    throw new IllegalArgumentException("Contains NAN");
                }
                total++;
            }
        }

    }
    public static void containsNAN(double[] arr) {
        boolean nan = false;
        int totalNAN = 0, total = 0;
        for (int i = 0; i < arr.length; i++) {
            if (Double.isNaN(arr[i])) {
                nan = true;
                totalNAN++;
                printArray(arr);
                throw new IllegalArgumentException("Contains NAN");
            }
            total++;
        }

    }
    public static void containsInfinity(double[] arr) {
        boolean nan = false;
        int totalNAN = 0, total = 0;
        for (int i = 0; i < arr.length; i++) {
            if (Double.isInfinite(arr[i])) {
                nan = true;
                totalNAN++;
                printArray(arr);
                throw new IllegalArgumentException("Contains Negative Infinity");
            }
            total++;
        }

    }
    public static void containsInfinity(double[][] arr){
        boolean nan = false;
        int totalNAN = 0, total = 0;
        for (int i = 0; i < arr.length ; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                if (Double.isInfinite(arr[i][j])){
                    nan = true;
                    totalNAN++;
                    printArray(arr);
                    throw new IllegalArgumentException("Contains Infinity");
                }
                total++;
            }
        }
    }
    public static void isOne(double[] arr){
        for (int i = 0 ; i < arr.length; i++){
            if ((int) (arr[i] * 10) == 10){
                printArray(arr);
                throw new IllegalArgumentException("Contains 1.0");
            }
        }
    }
    public static void isOne(double[][] arr){
        for (int i = 0; i < arr.length ; i++) {
            for (int j = 0 ; j < arr.length; j++){
                if (((int) (arr[i][j] * 10) == 10) && arr[i][j] <1.0000000001){
                    printArray(arr);
                    throw new IllegalArgumentException("Contains 1.0");
                }
            }
        }
    }
}
