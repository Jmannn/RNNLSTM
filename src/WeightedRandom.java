import java.util.NavigableMap;
import java.util.Random;
import java.util.TreeMap;
//https://stackoverflow.com/questions/6409652/random-weighted-selection-in-java
public class WeightedRandom {
    private final NavigableMap<Double, Character> map = new TreeMap<Double, Character>();
    private final Random random;
    private double total = 0;

    public WeightedRandom(char[] data, double[] prob) {
        this.random = new Random();
        for (int i = 0; i < data.length; i++) {
            add(prob[i],data[i]);
        }
    }

    public void add(double weight, Character result) {
        if (!(weight < 0)) {
            total += weight;
            map.put(total, result);
        } else {
            System.err.println("weight cannot be less than zero");
        }
    }

    public double pick() {
        double value = random.nextDouble() * total;
        System.err.println("Fails at pick method of WR");
        for (NavigableMap.Entry<Double, Character> entry : map.entrySet()) {
            System.out.println(entry.getKey() + ":" + entry.getValue().toString());
        }
        return map.higherEntry(value).getValue();
    }
}