

public class LTSM {

    /* These variables are for the backprop results. Only includes the extra
     * variables required for the backprop method that would normally be returned in single statement in
     * python.
     */
    double[][] fB, iB, cB, oB;
    double[] dfcs, dfhs;
    /* These variables are for the forwardProp results. Only includes the extra
     * variables required for the forwardProp method that would normally be returned in single statement in
     * python.
     */
    double[] fF;
    double[] iF;
    double[] cF;
    double[] oF;

    double[][] f, i, c, o, Gf, Gi, Gc, Go;
    double[] x, y, cs;
    int input, output, recurrences;
    double learningRate;
    public double[] error;

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
    //might pay to check this function
    public double[] sigmoid(double[] x){
        //90 percent sure input paramter x is an 1d array and returns a 1d array
        //np.exp(-x)) the negative in this line is a scalar
        //np.dot returns the lowest common dimensions 1d by 2d becomes 1d
        //sigmoid function does return an array becuase in python,
        // when you pass an array to a function, and you return it, it returns multiple
        // times to a new array.
        //JMath.isOne(x);
        JMath.containsNAN(x);
        double[] sigmoidMultiplied = new double[x.length];
        JMath.containsInfinity(x);
        for (int i = 0 ; i < x.length; i++){
            //1 / (1 + np.exp(-x))
            sigmoidMultiplied[i] = 1 / (1+Math.exp(-x[i]));
            //System.out.println(1-x[i]);
        }
        JMath.containsNAN(sigmoidMultiplied);
        JMath.containsInfinity(sigmoidMultiplied);
        //JMath.isOne(sigmoidMultiplied);
        return sigmoidMultiplied;
    }
    //check this function for the error
    public double[] dsigmoid(double[] x){
        double[] dsigmoidMultiplied = new double[x.length];
        JMath.containsNAN(x);
        JMath.containsInfinity(x);
        //JMath.printArray(x);
        for (int i = 0 ; i < x.length; i++){
            dsigmoidMultiplied[i] = 1 / (1+Math.exp(-x[i]));
        }
        JMath.containsInfinity(dsigmoidMultiplied);
        for (int i = 0 ; i < x.length; i++){
            dsigmoidMultiplied[i] *= 1 - dsigmoidMultiplied[i];
        }
        JMath.containsNAN(dsigmoidMultiplied);
        JMath.containsInfinity(dsigmoidMultiplied);
        return dsigmoidMultiplied;
    }
    public double[] tangent(double[] x){
        double[] tangentApplied = new double[x.length];
        JMath.containsNAN(x);
        JMath.containsInfinity(x);
        for (int i = 0 ; i < x.length; i++){
            tangentApplied[i] = Math.tanh(x[i]);
        }
        JMath.containsNAN(tangentApplied);
        JMath.containsInfinity(tangentApplied);
        return tangentApplied;
    }
    public double[] dtangent(double[] x){
        double[] dtangentApplied = new double[x.length];
        JMath.containsNAN(x);
        JMath.containsInfinity(x);
        for (int i = 0 ; i < x.length; i++){
            //1 - np.tanh(x)**2
            dtangentApplied[i] = 1 -(Math.pow(Math.tanh(x[i]), 2));
        }
        JMath.containsNAN(dtangentApplied);
        JMath.containsInfinity(dtangentApplied);
        return dtangentApplied;
    }

    public void forwardProp(){
        this.fF = sigmoid(JMath.dotProduct(this.f, this.x));
        JMath.containsNAN(this.fF);
        this.cs = JMath.dotProduct(this.f, this.cs);
        JMath.containsNAN(this.cs);
        //TODO: Happens after this point, maybe the this.i
        //System.out.println("this.i");
        //JMath.isOne(this.i);
        //JMath.isOne(this.x);
        //JMath.isOne(JMath.dotProduct(this.i, this.x));
        this.iF = sigmoid(JMath.dotProduct(this.i, this.x));
        //JMath.isOne(this.iF);
        //JMath.printArray(this.x);
        JMath.containsNAN(this.iF);
        this.cF = sigmoid(JMath.dotProduct(this.c, this.x));
        JMath.containsNAN(this.cF);
        this.cs = JMath.add1dArray(this.cs, JMath.dotProduct(this.iF,this.cF));
        JMath.containsNAN(this.cs);
        this.oF = sigmoid(JMath.dotProduct(this.o, this.x));
        JMath.containsNAN(this.oF);
        //Todo: this.o and this.cs both nan
        this.y = JMath.dotProduct(this.o, tangent(this.cs));
        JMath.containsNAN(this.y);
    }
    //it most likely comes from here
    public void backProp(double[] e, double[] pcs, double[] f, double[] i, double[] c, double[] o,
                         double[] dfcs, double[] dfhs){
        //do is doArr
        double[] doArr, dcs, dc, di, df, dpcs, dphs;
        double[][] ou, cu, iu, fu;
        JMath.containsNAN(e);
        JMath.containsNAN(pcs);
        JMath.containsNAN(f);
        JMath.containsNAN(i);
        JMath.containsNAN(c);
        JMath.containsNAN(o);
        JMath.containsNAN(dfcs);
        JMath.containsNAN(dfhs);

        JMath.containsInfinity(e);
        JMath.containsInfinity(pcs);
        JMath.containsInfinity(f);
        //System.out.println("heasdfasdf");
        //JMath.printArray(i);
        JMath.containsInfinity(i);
        JMath.containsInfinity(c);
        JMath.containsInfinity(o);
        JMath.containsInfinity(dfcs);
        JMath.containsInfinity(dfhs);

        e = JMath.clip(JMath.add1dArray(e,dfhs), 6, -6);
        doArr = JMath.dotProduct(tangent(this.cs),e);
        //second one is the one by 20, this.x
        ou = JMath.dotProduct(JMath.transpose(JMath.atleast2d(JMath.dotProduct(doArr, dtangent(o)))),
                JMath.atleast2d(this.x) );
        dcs = JMath.clip(JMath.add1dArray(JMath.dotProduct(e, JMath.dotProduct(o, dtangent(this.cs))),dfcs),
                6, -6);
        dc = JMath.dotProduct(dcs, i);
        cu = JMath.dotProduct(JMath.transpose(JMath.atleast2d(JMath.dotProduct(dc, dtangent(c)))),
                JMath.atleast2d(this.x));
        di = JMath.dotProduct(dcs, c);
        JMath.containsInfinity(i);
        JMath.containsInfinity(di);
        iu = JMath.dotProduct(JMath.transpose(JMath.atleast2d(JMath.dotProduct(di, dsigmoid(i)))),
                JMath.atleast2d(this.x));
        df = JMath.dotProduct(dcs, pcs);

        //TODO: here is nan  JMath.dotProduct(df, dsigmoid(f))

        //System.out.println(JMath.containsNAN( iu ));
        JMath.containsNAN(df);
        JMath.containsNAN( dsigmoid(f));
        JMath.containsInfinity(df);
        JMath.containsInfinity(f);
        JMath.containsInfinity(dsigmoid(f));
        JMath.containsNAN(JMath.dotProduct(df, dsigmoid(f)));
        JMath.containsNAN(JMath.atleast2d(JMath.dotProduct(df, dsigmoid(f))));
        JMath.containsNAN(JMath.transpose(JMath.atleast2d(JMath.dotProduct(df, dsigmoid(f)))));

        fu = JMath.dotProduct( JMath.transpose(JMath.atleast2d(JMath.dotProduct(df, dsigmoid(f)))), JMath.atleast2d(this.x) );
        JMath.containsNAN(fu);
        dpcs = JMath.dotProduct(dcs, f);

        //TODO: latest, is here fu and iu

        this.fB = fu;
        JMath.containsNAN(this.fB);
        this.iB = iu;
        JMath.containsNAN(this.iB);
        this.cB = cu;
        JMath.containsNAN(this.cB);
        this.oB = ou;
        JMath.containsNAN(this.oB);
        this.dfcs = dpcs;
        dphs = JMath.subArray(JMath.dotProduct(this.c, dc),0,this.output);
        dphs = JMath.add1dArray(JMath.subArray(JMath.dotProduct(this.o, doArr),0,this.output), dphs);
        dphs = JMath.add1dArray(JMath.subArray(JMath.dotProduct(this.i, di),0,this.output), dphs);
        dphs = JMath.add1dArray(JMath.subArray(JMath.dotProduct(this.f, df),0,this.output), dphs);
        this.dfhs = dphs;
        this.error = e;
        //in RNN : fu, iu, cu, ou, dfcs, dfhs
    }
    /*def update(self, fu, iu, cu, ou):
        #Update forget, input, cell, and output gradients
        self.Gf = 0.9 * self.Gf + 0.1 * fu**2
        self.Gi = 0.9 * self.Gi + 0.1 * iu**2
        self.Gc = 0.9 * self.Gc + 0.1 * cu**2
        self.Go = 0.9 * self.Go + 0.1 * ou**2

        #Update our gates using our gradients
        self.f -= self.learning_rate/np.sqrt(self.Gf + 1e-8) * fu
        self.i -= self.learning_rate/np.sqrt(self.Gi + 1e-8) * iu
        self.c -= self.learning_rate/np.sqrt(self.Gc + 1e-8) * cu
        self.o -= self.learning_rate/np.sqrt(self.Go + 1e-8) * ou
        return */
    public void  update(double[][] fu, double[][] iu, double[][] cu, double[][] ou){
        JMath.containsNAN(fu);
        JMath.containsNAN(iu);
        JMath.containsNAN(cu);
        JMath.containsNAN(ou);
        this.Gf = JMath.add2dArray(JMath.multiply2d(this.Gf, 0.9),
                JMath.multiply2d(JMath.power2d(fu, 2), 0.1));

        this.Gi = JMath.add2dArray(JMath.multiply2d(this.Gi, 0.9),
                JMath.multiply2d(JMath.power2d(iu, 2), 0.1));
        this.Gc = JMath.add2dArray(JMath.multiply2d(this.Gc, 0.9),
                JMath.multiply2d(JMath.power2d(cu, 2), 0.1));
        this.Go = JMath.add2dArray(JMath.multiply2d(this.Go, 0.9),
                JMath.multiply2d(JMath.power2d(ou, 2), 0.1));

        this.f = JMath.sub2dArray(this.f, JMath.dotProduct(JMath.divideByArray(this.learningRate,
                JMath.power2d(JMath.add2d(this.Gf, 1e-8), 0.5)), fu) );
        this.i = JMath.sub2dArray(this.i, JMath.dotProduct(JMath.divideByArray(this.learningRate,
                JMath.power2d(JMath.add2d(this.Gi, 1e-8), 0.5)), iu) );
        //System.out.println(JMath.containsNAN(this.i));
        this.c = JMath.sub2dArray(this.c, JMath.dotProduct(JMath.divideByArray(this.learningRate,
                JMath.power2d(JMath.add2d(this.Gc, 1e-8), 0.5)), cu) );
        this.o = JMath.sub2dArray(this.o, JMath.dotProduct(JMath.divideByArray(this.learningRate,
                JMath.power2d(JMath.add2d(this.Go, 1e-8), 0.5)), ou) );
    }
}
