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
    LTSM lstm;
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
        this.lstm = new LTSM(input,output,recurrences,learningRate);
    };
    public double[] sigmoid(double[] x){
        //90 percent sure input paramter x is an 1d array and returns a 1d array
        //np.exp(-x)) the negative in this line is a scalar
        //np.dot returns the lowest common dimensions 1d by 2d becomes 1d
        //sigmoid function does return an array becuase in python,
        // when you pass an array to a function, and you return it, it returns multiple
        // times to a new array.
        double[] sigmoidMultiplied = new double[x.length];
        for (int i = 0 ; i < x.length; i++){
            sigmoidMultiplied[i] = 1 / (1-x[i]);
        }
        return sigmoidMultiplied;
    }

    public double[] dsigmoid(double[] x){
        double[] dsigmoidMultiplied = new double[x.length];
        for (int i = 0 ; i < x.length; i++){
            dsigmoidMultiplied[i] = 1 / (1-x[i]);
        }
        for (int i = 0 ; i < x.length; i++){
            dsigmoidMultiplied[i] *= 1 - dsigmoidMultiplied[i];
        }
        return dsigmoidMultiplied;
    }
    public double[][] forwardProp(){

        //since java cannot return multiple variables at
        // once, you will need to use call the forward prop, and then use getters

        double[] cs, hs; //these come from the main class variables
        double[] f, inp,c, o; // these come from the forward prop only variables

        for(int i = 1; i < recurrences+1; i++){
            lstm.x = JMath.hStack(this.ha[i-1], this.x);
            lstm.forwardProp();
            cs = lstm.cs;
            hs = lstm.hs;
            f = lstm.fFP;
            inp = lstm.iFP;
            c = lstm.cFP;
            o = lstm.oFP;

            this.ca[i] = cs;
            this.ha[i] = hs;
            this.af[i] = f;
            this.ai[i] = inp;
            this.ac[i] = c;
            this.ao[i] = o;
            this.oa[i] = sigmoid(JMath.dotProduct(this.w, hs));
            this.x = this.expectedOutput[i-1];

        }
        return this.oa;
    }

    public double backProp(){

        double totalError = 0.0;
        double[] error = new double[];
        double[] dfcs = new double[output];
        double[] dfhs = new double[output];
        double[][] tu = new double[output][output];
        double[][] tfu = new double[output][input+output];
        double[][] tiu = new double[output][input+output];
        double[][] tcu = new double[output][input+output];
        double[][] tou = new double[output][input+output];

        for (int i = this.recurrences; i > -1; i--){
            error = JMath.difference(this.oa[i], this.expectedOutput[i]);
            // this 2d dot product
            tu = JMath.add2dArray(tu, JMath.dotProduct(JMath.atleast2d(
                    JMath.dotProduct(error , dsigmoid(this.oa[i])) ), JMath.transpose(JMath.atleast2d(ha[i]))));

        }

        return totalError;



    }


}
