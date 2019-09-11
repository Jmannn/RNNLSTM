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
            hs = lstm.y;
            f = lstm.fF;
            inp = lstm.iF;
            c = lstm.cF;
            o = lstm.oF;

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
        double[] error = new double[output];
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
            error = JMath.dotProduct(this.w, error);
            lstm.x = JMath.hStack(this.ha[i-1], this.ia[i]);
            lstm.cs = ca[i];

            lstm.backProp(error, this.ca[i-1], this.af[i], this.ai[i], this.ac[i], this.ao[i], dfcs, dfhs);

            tfu = JMath.add2dArray(tfu,lstm.fB);

            tiu = JMath.add2dArray(tfu,lstm.iB);

            tcu = JMath.add2dArray(tfu,lstm.cB);

            tou = JMath.add2dArray(tfu,lstm.oB);


        }
        this.lstm.update(JMath.divide2d(tcu, this.recurrences),JMath.divide2d(tfu, this.recurrences),
                JMath.divide2d(tou, this.recurrences),JMath.divide2d(tiu, this.recurrences));
        update(JMath.divide2d(tu, this.recurrences));

        return totalError;



    }
    public void update(double[][] u){
        this.g = JMath.add2dArray(JMath.multiply2d(this.g, 0.95), JMath.multiply2d(JMath.power2d(u, 2), 0.1));
        //self.w -= self.learning_rate/np.sqrt(self.G + 1e-8) * u
        this.w = JMath.sub2dArray(this.w, JMath.dotProduct( JMath.dDivide(JMath.power2d(JMath.add2d(this.g,1e-8), 0.5),this.learningRate), u   ));
    }
    public double[][]  sample(){
        double[] cs, hs; //these come from the main class variables
        double[] f, inp,c, o; // these come from the forward prop only variables
        for (int i = 1; i < this.recurrences+1; i++){
            int maxI;
            //self.LSTM.x = np.hstack((self.ha[i-1], self.x))
            this.lstm.x = JMath.hStack(this.ha[i-1], this.x);

            lstm.forwardProp();
            cs = lstm.cs;
            hs = lstm.y;
            f = lstm.fF;
            inp = lstm.iF;
            c = lstm.cF;
            o = lstm.oF;

            maxI = JMath.argMax(this.x);

            this.x = new double[this.x.length];
            this.x[maxI] = 1; // eval this by looking for a greater than, all the rest should be 0 but the value wont be exactly one
        }

    }


}
