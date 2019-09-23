import java.util.Random;

public class RNNApp {
    public static void main(String[] args) {
        System.out.println("Initialise Hyperparameters");
        int iterations = 20;
        double learningRate = 0.001;
        System.out.println("Reading I/O from textfile");
        TextFileIO io = new TextFileIO("/home/johnny/IdeaProjects/RNN/resources/test.txt");

        System.out.println("Reading from disk success! Proceeding with training.");
        RNN rnn = new RNN(io.getUniqueWordsNum(),io.getUniqueWordsNum(), io.getOutputSize(),
                io.getOutput(), learningRate);

        double error = 0;
        double[] seed;
        double[][] output;
        int randomSeedInt = 0;
        Random ran = new Random();
        //Todo: after first iteration, all the values seem to get wipped from rnn.sample()
        for (int i = 1; i < iterations; i++) {
            rnn.forwardProp();
            error = rnn.backProp();
            System.out.println("Error for iteration i: "+ i + " : " + error);
            //JMath.printArray(rnn.sample());
            if ((error > -10 && error < 10) || (i % 10 == 0)){
                seed = new double[rnn.x.length];
                randomSeedInt = ran.nextInt(seed.length);
                //selects the word basically, by denoting a 1
                seed[randomSeedInt] = 1;
                rnn.x = seed;
                output = rnn.sample();
                io.export(output,io.getData(),"/home/johnny/IdeaProjects/RNN/resources/out.txt");

            }
        }

    }
}