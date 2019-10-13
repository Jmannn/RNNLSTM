import java.util.Random;
public class WeightedRandom {

    private double total;
    private char[] data;
    private double[] probWheel;
    private Random ran;

    public WeightedRandom(char[] data, double[] prob) {
        this.ran = new Random();
        this.data = data;
        this.total = 0;
        this.probWheel = new double[prob.length];
        //JMath.containsNAN(prob);
        for (int i = 0; i < prob.length; i++) {
            this.total += prob[i];
            this.probWheel[i] = this.total;
        }

    }


    public char pick() {
        char pick;
        //rangeMin + (rangeMax - rangeMin) * r.nextDouble();
        double choice = this.total * ran.nextDouble();
        for (int i = 0; i < probWheel.length-1; i++) {

            if(probWheel[i]> choice){

                return this.data[i];
            }
        }
        return this.data[this.data.length-1];
    }
}