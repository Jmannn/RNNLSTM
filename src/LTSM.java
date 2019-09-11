public class LTSM {

    /* These variables are for the backprop results. Only includes the extra
     * variables required for the backprop method that would normally be returned in single statement in
     * python.
     */
    double[][] fB, iB, cB, oB;
    /* These variables are for the forwardProp results. Only includes the extra
     * variables required for the forwardProp method that would normally be returned in single statement in
     * python.
     */
    double[] fF;
    double[] iF;
    double[] cF;
    double[] oF;

    double[][] f, i, c, o, dfcs, dfhs, Gf, Gi, Gc, Go;
    double[] x, y, cs;
    int input, output, recurrences;
    double learningRate;

    /* Constructor for the ltsm class that initializes some variables. */
    public LTSM(int input, int output, int recurrences, double learningRate){
        this.x = new double[input+output];
        this.input = input + output;
        this.y = new double[output];
        this.output = output;
        this.cs = new double[output];
        this.recurrences = recurrences;
        this.learningRate = learningRate;
        //self.f = np.random.random((output, input+output))
        this.f = JMath.randomSample2D(input, input+output);
        this.i = JMath.randomSample2D(input, input+output);
        this.c = JMath.randomSample2D(input, input+output);
        this.o = JMath.randomSample2D(input, input+output);

        this.Gf = new double[input][input+output];
        this.Gi = new double[input][input+output];
        this.Gc = new double[input][input+output];
        this.Go = new double[input][input+output];
    }
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
    public double[] tangent(double[] x){
        double[] tangentApplied = new double[x.length];
        for (int i = 0 ; i < x.length; i++){
            tangentApplied[i] = Math.tanh(x[i]);
        }

        return tangentApplied;
    }
    public double[] dtangent(double[] x){
        double[] dtangentApplied = new double[x.length];
        for (int i = 0 ; i < x.length; i++){
            //1 - np.tanh(x)**2
            dtangentApplied[i] = 1 -(Math.pow(Math.tanh(x[i]), 2));
        }

        return dtangentApplied;
    }




    public void forwardProp(){
        /*
        self.y = o * self.tangent(self.cs)
        return self.cs, self.y, f, i, c, o */
        this.fF = sigmoid(JMath.dotProduct(this.f, this.x));
        this.cs = JMath.dotProduct(this.f, this.cs);
        this.iF = sigmoid(JMath.dotProduct(this.i, this.x));
        this.cF = sigmoid(JMath.dotProduct(this.c, this.x));
        this.cs = JMath.add1dArray(this.cs, JMath.dotProduct(this.iF,this.cF));
        this.oF = sigmoid(JMath.dotProduct(this.o, this.x));
        this.y = JMath.dotProduct(this.o, tangent(this.cs));
    }
    public void backProp(){

    }
}
