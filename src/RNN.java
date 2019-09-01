public class RNN {

    // timesteps is like number of pixels in a picture,

    /* Class members */

    /* Initial input */
    double[] x;
    /* Expected output, shifted by a timestep */
    double[] y;
    /* Input size */
    int input;
    /* Output Size */
    int output;
    /* Weight matrix for interpreting results from the hidden weight matrix */
    double[][] w;
    /*Matrix used in RMSprop to decay the learning rate */
    double[][] g;
    /* Number of recurrences i.e number of inputs , number of chars*/
    int recurrences;
    /* Learing rate */
    double learningRate;
    /* Array for storing inputs over recurrences */
    double[][] ia;
    /* For storing cell states */
    double[][] ca;
    /* For storing outputs */
    double[][] oa;
    /* For storing hidden states */
    double[][] ha;
    /* Forget gate */
    double[][] af;
    /* Input gate */
    double[][] ai;
    /* Cell state*/
    double[][] ac;
    /* Output gate */
    double[][] ao;
    /* Expected output array 2d */
    double[][] expectedOutput;
    /* LTSM */
    LTSM ltsm;
    RNN(int input, int output, int recurrences, double[][] expectedOutput ,double learningRate){
        //.T means to transpose
        this.x = new double[input];
        this.input = input;
        this.y = new double[output];
        this.output = output;
        this.w = JMath.randomSample2D(output, output);
        this.g = new double[output][output];
        this.recurrences = recurrences;
        this.learningRate = learningRate;
        this.ia = new double[recurrences+1][input];
        this.ca = new double[recurrences+1][output];
        this.oa = new double[recurrences+1][output];
        this.ha = new double[recurrences+1][output];
        this.af = new double[recurrences+1][output];
        this.ai = new double[recurrences+1][output];
        this.ac = new double[recurrences+1][output];
        this.ao = new double[recurrences+1][output];
        //self.expected_output = np.vstack((np.zeros(expected_output.shape[0]), expected_output.T))
        // expected_output.shape[0] = expectedOutput.length
        //vstack just stacks em together by the row or fist []
        this.expectedOutput = JMath.vStack(new double[1][expectedOutput.length], JMath.transpose(expectedOutput));
        this.ltsm = new LTSM(input,output,recurrences,learningRate);
    };
    public static double sigmoid(){
        //90 percent sure input paramter x is an 1d array and returns a 1d array
        //np.exp(-x)) the negative in this line is a scalar
        //np.dot returns the lowest common dimensions 1d by 2d becomes 1d
    }

}
