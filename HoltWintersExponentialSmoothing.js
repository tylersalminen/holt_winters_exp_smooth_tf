/* eslint-disable id-length */

// https://github.com/antoinevastel/zodiac-ts/blob/master/index.js#L519

class HoltWintersExponentialSmoothing {

    constructor(
        $endogenous,
        $parameters,
        $abortController
    ){

        this.$endogenous = tf.tensor1d(($endogenous instanceof Float32Array) ? $endogenous : new Float32Array($endogenous));
        this.$parameters = {
            auto: false,

            optimizer: 0,
            learningRate: 0.1, /// learning rate 
            batchSize: 2048,
            epochs: 20,
            regularizerAlpha: 0.5,

            seasonalPeriods: 0,
            seasonalProcess: 1, /// 0 | 1 | 2
            trendProcess: 1, /// 0 | 1 | 2
            dampenProcess: 1, /// 0 | 1

            initialValuesGenerationType: 1,

            alpha: 0,
            beta: 0,
            gamma: 0,
            phi: 1, /// 1 === no dampening
            alphaBounds: [0, 1],
            betaBounds: [0, 1],
            gammaBounds: [0, 1],
            phiBounds: [0, 1],
            removeBias: false,
            ...$parameters
        };
        this.$abortController = $abortController;

        /// force dampening off if no trend
        if(this.$parameters.trendProcess === 0){

            this.$parameters.dampenProcess = 0;
            this.$parameters.phi = 1;

        }

        this.$alpha = this.$parameters.alpha;
        this.$gamma = this.$parameters.gamma;
        this.$beta = this.$parameters.beta;
        this.$phi = this.$parameters.phi;
        this.$observations = $endogenous.length;
        this.$damped = this.$parameters.dampenProcess !== 0;
        this.$fitRecord = [];
        this.$predictRecord = [];

        if(
            this.$parameters.seasonalProcess !== 0 &&
            this.$parameters.seasonalPeriods <= 0
        ){

            throw new Error("seasonalPeriods should be > 0");

        }

        if(
            this.$parameters.seasonalProcess === 2 ||
            this.$parameters.trendProcess === 2
        ){

            for(let i = 0; i < $endogenous.length; i++){

                if($endogenous[i] <= 0){

                    throw new Error("endogenous data must be strictly positive if season or trend positive is multiplicative");

                }

            }

        }
        
        this.$$checkParameter(this.$parameters.alpha);
        this.$$checkParameter(this.$parameters.beta);
        this.$$checkParameter(this.$parameters.gamma);
        this.$$checkParameter(this.$parameters.phi);

        this.$$checkParameterBoundary(this.$parameters.alphaBounds);
        this.$$checkParameterBoundary(this.$parameters.betaBounds);
        this.$$checkParameterBoundary(this.$parameters.gammaBounds);
        this.$$checkParameterBoundary(this.$parameters.phiBounds);

        


        Object.seal(this);


    }

    
    $$checkParameter ($parameter){

        if($parameter < 0 || $parameter > 1){

            throw new Error("invalid parameter value ");

        }


    }

    $$checkParameterBoundary ($bounds){

        if(
            !Array.isArray($bounds) || 
            $bounds.length !== 2
        ){

            throw new Error("Bounds must be an array of length 2");

        }

        this.$$checkParameter($bounds[0]);
        this.$$checkParameter($bounds[1]);

        if(
            $bounds[0] > $bounds[1]
        ){

            throw new Error("Invalid parameter boundaries");

        }

    }

    $$incrementBoundary(
        $parameter,
        $bounds
    ) {

        $parameter[0] += this.$parameters.learningRate;

        return $parameter[0] <= $bounds[1];

    }

    * $$iterateParameterSpaceBatched() {

        const $alphaBounds = this.$parameters.alphaBounds;
        const $betaBounds = this.$parameters.trendProcessProcess === 0 ? [this.$beta, this.$beta] : this.$parameters.betaBounds;
        const $gammaBounds = this.$parameters.seasonalProcess === 0 ? [this.$gamma, this.$gamma] : this.$parameters.gammaBounds;
        const $phiBounds = this.$parameters.dampenProcess === 0 ? [this.$phi, this.$phi] : this.$parameters.phiBounds;

        const $alpha = [$alphaBounds[0]];
        const $beta = [$betaBounds[0]];
        const $gamma = [$gammaBounds[0]];
        const $phi = [$phiBounds[0]];

        /// create compact batch array of 3 length tuples
        const $batch = [];

        for(let i = 0; i < this.$parameters.batchSize; i++){

            $batch.push(
                [
                    /// initialize to floats
                    NaN,
                    NaN,
                    NaN,
                    NaN
                ]
            );

        }

        let i = -1;

        do {
            
            do {

                do {
                

                    do {

                        i++;

                        /// send prior batch first
                        if(i === $batch.length){

                            yield $batch;

                            i = 0; /// reset i to 0

                        }

                        $batch[i][0] = $alpha[0];
                        $batch[i][1] = $beta[0];
                        $batch[i][2] = $gamma[0];
                        $batch[i][3] = $phi[0];
                        

                    }
                    while(this.$$incrementBoundary($phi, $phiBounds));

                    $phi[0] = $phiBounds[0];

                }
                while(this.$$incrementBoundary($beta, $betaBounds));

                $beta[0] = $betaBounds[0];

            }
            while(this.$$incrementBoundary($gamma, $gammaBounds));

            $gamma[0] = $gammaBounds[0];

        }
        while(this.$$incrementBoundary($alpha, $alphaBounds));

    
        /// yield rest of batch
        yield $batch.slice(0, i);

        
        

    }

    
    $$doTrainingPredict (
        ...$parameters
    ) {

        const [
            $fitted,
            $residuals,
            $sse,
            $mse,
            $k,
            $aic,
            $aicc,
            $bic
        ] = this.predict(
            0,
            ...$parameters,
            {
                recordOutputForSummary: false,
                //calculateInformationCriterion: false, /// don't need aic, bic, etc
                removeBias: false, /// don't need bias removed
                initialValuesGenerationType: 0
            }
        );

        /// dispose non recorded data
        $fitted.dispose();
        $residuals.dispose();
        $sse.dispose();
        $k.dispose();

        /// keep stats, dispose() class method cleans them up
        tf.keep($mse);
        tf.keep($aic);
        tf.keep($aicc);
        tf.keep($bic);

        this.$fitRecord.push(
            [   
                /// clone parameters since they are variables
                $parameters[0] ? tf.keep($parameters[0].clone().clipByValue(...this.$parameters.alphaBounds)) : undefined,
                $parameters[1] ? tf.keep($parameters[1].clone().clipByValue(...this.$parameters.betaBounds)) : undefined,
                $parameters[2] ? tf.keep($parameters[2].clone().clipByValue(...this.$parameters.gammaBounds)) : undefined,
                $parameters[3] ? tf.keep($parameters[3].clone().clipByValue(...this.$parameters.phiBounds)) : undefined,
                $mse,
                $aic,
                $aicc,
                $bic
            ]
        );

        return $mse;

    }

    async $$fitBruteForce () {
        

        /// get permutations of all 3 parameters in batches of n
        /// perhaps scale batch size according to size of endogenous array to keep consistent memory usage?
        const $bestParameterSet = [
            Infinity,
            NaN,
            NaN,
            NaN,
            NaN
        ];

        for(const $parameters of this.$$iterateParameterSpaceBatched()){
            

            const $parametersTensor = tf.tensor2d($parameters);
            const $parametersSplit = $parametersTensor.split(4, 1);
            
            /// shape into column vectors
            /// [ [a1], [a2], .... [an] ]
            /// [ [b1], [b2], .... [bn] ]
            /// [ [g1], [g2], .... [gn] ]

            /// redict with no forecast, only need to compute SSE which only makes sense over endogenous data
            const $mse = this.$$doTrainingPredict(
                ...$parametersSplit
            );

            /// get mse and take opportunity to check for cancellation
            // eslint-disable-next-line no-await-in-loop
            const $errors = await $mse.array();

            this.$abortController.checkCancelled();
            
            for(let i = 0; i < $errors.length; i++){

                const $error = $errors[i];

                if($error < $bestParameterSet[0]){

                    const $chunkParameters = $parameters[i];

                    $bestParameterSet[0] = $error;
                    $bestParameterSet[1] = $chunkParameters[0];
                    $bestParameterSet[2] = $chunkParameters[1];
                    $bestParameterSet[3] = $chunkParameters[2];
                    $bestParameterSet[4] = $chunkParameters[3];
                    
                }

            }

        }


        this.$alpha = $bestParameterSet[1];
        this.$beta = $bestParameterSet[2];
        this.$gamma = $bestParameterSet[3];
        this.$phi = $bestParameterSet[4];

    }

    async $$fitEpochs () {
    
        // https://www.geeksforgeeks.org/tensorflow-js-tf-train-optimizer-class-minimize-method/

        let $alpha = undefined;
        let $beta = undefined;
        let $gamma = undefined;
        let $phi = undefined;
        let $optimizer = undefined;

        try {

            const $variableList = [];

            // Create a variables for each parameter
            $alpha = tf.variable(tf.tensor2d([[this.$alpha]]));
            $variableList.push($alpha);

            if(this.$parameters.trendProcess !== 0){
                $beta = tf.variable(tf.tensor2d([[this.$beta]]));
                $variableList.push($beta);
            }

            if(this.$parameters.seasonalProcess !== 0){
                $gamma = tf.variable(tf.tensor2d([[this.$gamma]]));
                $variableList.push($gamma);
            }

            if(this.$parameters.dampenProcess !== 0){
                $phi = tf.variable(tf.tensor2d([[this.$phi]]));
                $variableList.push($phi);
            }

            const $optimizerName = {
                0: "adam",
                1: "sgd",
                2: "adamax"
            }[this.$parameters.optimizer - 1];
            
            // Define the optimizer
            $optimizer = tf.train[$optimizerName](this.$parameters.learningRate);

            // Training loop
            for (let i = 0; i < this.$parameters.epochs; i++) { /// epochs

                $optimizer.minimize(
                    () => {


                        /// is within tidy() callback implicitly ....

                        /// redict with no forecast, only need to compute SSE which only makes sense over endogenous data
                        let $mse = this.$$doTrainingPredict(
                            $alpha,
                            $beta,
                            $gamma,
                            $phi
                        );

                        /// adjust loss function if variables are approaching clipped bounds using a "regularlizer" but one bounded via clipping
                        /// https://stackoverflow.com/questions/62236460/how-to-set-bounds-and-constraints-on-tensorflow-variables-tf-variable
                        /// loss += alpha*tf.abs(tf.math.tan(((x-5.5)/4.5)*pi/2))
                        if(this.$parameters.regularizerAlpha !== 0){

                            /// penalize parameters that are at bounds, exponentially approaches
                            /// inifinty at boundaries
                            $mse = $mse.add(
                                this.$$variableRegularizerLossRate(
                                    $alpha,
                                    this.$parameters.alphaBounds
                                )
                            );

                            if($beta){
                            
                                $mse = $mse.add(
                                    this.$$variableRegularizerLossRate(
                                        $beta,
                                        this.$parameters.betaBounds
                                    )
                                );

                            }

                            if($gamma){
                            
                                $mse = $mse.add(
                                    this.$$variableRegularizerLossRate(
                                        $gamma,
                                        this.$parameters.gammaBounds
                                    )
                                );

                            }

                            if($phi){
                            
                                $mse = $mse.add(
                                    this.$$variableRegularizerLossRate(
                                        $phi,
                                        this.$parameters.phiBounds
                                    )
                                );

                            }

                        }
                        
                        return $mse.squeeze();


                    },
                    false,
                    $variableList
                );
        
                // Print the current value of x and the loss
                
                // eslint-disable-next-line no-await-in-loop
                await tf.nextFrame(); // To prevent freezing the UI

                /// check for cancellation
                this.$abortController.checkCancelled();

            }

            [
                this.$alpha,
                this.$beta,
                this.$gamma,
                this.$phi
            ] = await Promise.all([
                $alpha.clipByValue(...this.$parameters.alphaBounds).squeeze().array(),
                $beta?.clipByValue(...this.$parameters.betaBounds).squeeze().array(),
                $gamma?.clipByValue(...this.$parameters.gammaBounds).squeeze().array(),
                $phi?.clipByValue(...this.$parameters.phiBounds).squeeze().array()
            ]);

            /// check for cancellation
            this.$abortController.checkCancelled();

        }
        finally {

            $alpha?.dispose();
            $beta?.dispose();
            $gamma?.dispose();
            $phi?.dispose();
            $optimizer?.dispose();

        }

    }

    $$variableCenter (
        $bounds
    ) {

        return ($bounds[1] / 2) + $bounds[0];

    }
    
    $$variableRegularizerLossRate (
        $variable,
        $bounds
    ) {

        const $boundsCenter = this.$$variableCenter($bounds);
        
        return tf.abs(
            tf.tan(
                /// this fn is asymtotic at the parameter boundaries but is defined outside
                /// of them so need to clip to boudnaries if exceeds them
                $variable.clipByValue(...$bounds).sub($boundsCenter + $bounds[0])
                .div($boundsCenter)
                .mul(
                    Math.PI / 2
                )
            )
        ).mul(
            this.$parameters.regularizerAlpha /// alpha
        );

    }
    
    async fit () {

        if(
            this.$parameters.optimizer === 0
        ){

            await this.$$fitBruteForce();

        }
        else if(
            this.$parameters.optimizer === 1 ||
            this.$parameters.optimizer === 2 ||
            this.$parameters.optimizer === 3
        ){

            await this.$$fitEpochs();

        }

    }

    

    // https://js.tensorflow.org/api/latest/?_gl=1*1uup375*_ga*MTM1MTU5MjExLjE2OTcwNTA2MjA.*_ga_W0YLR4190T*MTcxMjgzNDk3My4yMi4xLjE3MTI4MzQ5NzQuMC4wLjA.#mod

    $$initialValues (
        $type
    ) {

        /// 0: optimizing
        /// 1: estimate
        /// 2: heuristic
        return this.$$initialValuesSimple();

        switch($type){
            case 0:

                return this.$$initialValuesSimple();

            case 1:

                if (this.$observations < 10 + 2 * (this.$parameters.seasonalPeriods / 2)){

                    return this.$$initialValuesSimple();

                }
                    
                return this.$$initialValuesHeuristic();
            
            case 2:

                return this.$$initialValuesHeuristic();

        }

    }


    $$initialValuesHeuristic () {

        const $endogenous = this.$endogenous;
        const $seasonalPeriods = this.$parameters.seasonalPeriods;
        const $observations = this.$observations;

        if($endogenous.length < 10){

            throw new Error("initials cannot be generated with heuristic if < 10 observations");

        }

        if (this.$parameters.seasonalProcess === 0){


        }
        else {

            if($observations < 2 * $seasonalPeriods){

                throw new Error("there must be at least 2 full seasons of observations to compute hueristic initializers with seasonal process");

            }

            const $minimumObservations = 10 + (2 * $seasonalPeriods);

            if($observations < $minimumObservations){

                throw new Error("there must be at least 2 full seasons + 10 observations of observations to compute hueristic initializers with seasonal process");

            }

            let $kCycles = Math.min(5, $seasonalPeriods); // min(5, nobs // seasonal_periods)
            $kCycles = Math.max($kCycles, Math.ceil($minimumObservations / $seasonalPeriods)); // max(k_cycles, int(np.ceil(min_obs / seasonal_periods)))

            /// pd.Series(endog[:seasonal_periods * k_cycles])
            const $series = $endogenous.slice(
                [0],
                [$seasonalPeriods * $kCycles]
            );

            
        }

    }

    $$initialValuesSimple () {

        const $endogenous = this.$endogenous;

        let $level = undefined;
        let $trend = undefined;
        let $seasonals = undefined;

        /// if has no seasonal process
        if (this.$parameters.seasonalProcess === 0){
            
            $level = $endogenous.gather([0]); // y[0]

            if(this.$parameters.trendProcess !== 0){

                /// is additive trend process
                if(this.$parameters.trendProcess === 1){
                    
                    $trend = $endogenous.gather([1]).sub($endogenous.gather([0])); //  y[1] - y[0] 

                }
                /// is multiplicitive trend process
                else if(this.$parameters.trendProcess === 2){

                    $trend = $endogenous.gather([1]).div($endogenous.gather([0])); //  y[1] / y[0] 

                }

            }
            
        }
        else {

            if(this.$observations.length < 2 * this.$parameters.seasonalPeriods){

                throw new Error("there must be at least 2 full seasons of observations to compute simple initializers");

            }
                    
            /// for level get mean of first period of every season
            /// y[np.arange(self.nobs) % m == 0].mean()
            $level = $endogenous.gather(
                tf.range(0, this.$observations, this.$parameters.seasonalPeriods, "int32")
            ).mean();

            /// if trend process exists then extract lead and lag
            /// lead, lag = y[m : m + m], y[:m]

            if(this.$parameters.trendProcess !== 0){


                const $trendLead = $endogenous.slice(
                    this.$parameters.seasonalPeriods,
                    this.$parameters.seasonalPeriods
                );

                const $trendLag = $endogenous.slice(
                    0,
                    this.$parameters.seasonalPeriods
                );

                /// is additive trend process
                /// b0 = ((lead - lag) / m).mean()
                if(this.$parameters.trendProcess === 1){

                    $trend = $trendLead.sub($trendLag).div(this.$parameters.seasonalPeriods).mean();
                    
                }
                /// is multiplicitive trend process
                /// b0 = np.exp((np.log(lead.mean()) - np.log(lag.mean())) / m)
                else if(this.$parameters.trendProcess === 2){

                    $trend = $trendLead.mean().log().sub($trendLag.mean().log()).div(this.$parameters.seasonalPeriods).exp();
                    //throw new Error("Not implemented");
                    
                }


            }

            const $seasonLag = $endogenous.slice(
                0,
                this.$parameters.seasonalPeriods
            );
            
            /// if has additive seasonal process
            /// list(y[:m] - l0)
            if(
                this.$parameters.seasonalProcess === 1
            ){

                
                $seasonals = $seasonLag.sub(
                    $level
                );

            }
            /// if has multiplicitive seasonal process
            /// list(y[:m] / l0)
            else if (
                this.$parameters.seasonalProcess === 2
            ){


                $seasonals = $seasonLag.div(
                    $level
                );

            }

        }

        return [
            $level,
            $trend,
            $seasonals
        ];


    }

    predictInterval(){

        /// models like following have exact solutions to PI so don't need to simulate 
        /// Trend: none, Season: none, ([A]NN)
        /// Trend: Add, Dampended, Seasons: Add ([A]AdA)
        /// Trend: Add, Seasons: Add ([A]AA) 

        // https://stackoverflow.com/questions/70277316/how-to-take-confidence-interval-of-statsmodels-tsa-holtwinters-exponentialsmooth 
        // https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/exponential_smoothing/ets.py#L2101

        /// do simiulations logic (THIS IS DIFFERENT THAN PREDICT)

    }

    predict (
        $predictions, 
        $alpha, 
        $beta, 
        $gamma,
        $phi,
        $options
    ) {

        return tf.tidy(() => {

            $options = {
                recordOutputForSummary: true,
                calculateInformationCriterion: true, /// don't need aic, bic, etc
                ...this.$parameters, /// don't need bias removed
                ...$options
            };

            /// localize
            const $observations = this.$observations;
            const $trendProcess = $options.trendProcess;
            const $seasonalProcess = $options.seasonalProcess;
            const $seasonalPeriods = $options.seasonalPeriods;
            const $removeBias = $options.removeBias;
            const $zeroScalar = tf.scalar(0);
            const $oneScalar = tf.scalar(1);
            const $twoScalar = tf.scalar(2);
            const $threeScalar = tf.scalar(3);
            const $endogenous = this.$endogenous;
            const $damped = this.$damped;

            /// scalars
            const $observationss = tf.scalar($observations);

            /// BOX COX .... do box cox if needed on endogenous data

            /// get initial values
            const [
                $initialLevel,
                $initialTrend,
                $initialSeasonals
            ] = this.$$initialValues(
                $options.initialValuesGenerationType
            );

            /// resolve an calc smoothing parameters
            /// variables
            const $alphas = ($alpha ?? tf.tensor2d([[this.$alpha]])).clipByValue(...$options.alphaBounds);
            const $alphaV = $alphas.transpose();
            const $betas = ($beta ?? tf.tensor2d([[this.$beta]])).clipByValue(...$options.betaBounds);
            const $betaV = $betas.transpose();
            const $gammas = ($gamma ?? tf.tensor2d([[this.$gamma]])).clipByValue(...$options.gammaBounds);
            const $gammaV = $gammas.transpose();

            const $batchSize = $alphas.size;
            const $phis = $phi ?? tf.tensor2d([[this.$phi]]); // phi if damped else 1.0
            const $phiV = $phis.transpose(); // phi if damped else 1.0
            
            /// 
            const $alphaCs = $oneScalar.sub($alphas);
            const $alphaCV = $alphaCs.transpose();
            const $betaCs = $oneScalar.sub($betas);
            const $betaCV = $betaCs.transpose();
            const $gammaCs = $oneScalar.sub($gammas);
            const $gammaCV = $gammaCs.transpose();

            const $yAlpha = $endogenous.mul($alphas); /// y_alpha[:] = alpha * y
            const $yGamma = $seasonalProcess ? $endogenous.mul($gammas) : tf.zeros([$observations]); /// p.zeros((self.nobs,))

            /// create / initialize levels, trends, and seasonals, convert to array of scalars
            /// so can set values index wise
            const $levels = new Array($observations + $predictions + 1); // np.zeros((self.nobs + h + 1,))
            const $trends = new Array($observations + 1); // np.zeros((self.nobs + h + 1,))
            $trends.fill($zeroScalar);
            const $seasonals = new Array($observations + $seasonalPeriods); // np.zeros((self.nobs + h + m + 1,))

            /// initialize first periods of vectors
            $levels[0] = $initialLevel.reshape([1]).tile([$batchSize]); /// is scalar, convert to 1d
            if($trendProcess !== 0){
                $trends[0] = $initialTrend.reshape([1]).tile([$batchSize]); /// is scalar, convert to 1d
            }
            if($seasonalProcess !== 0){
                $seasonals.splice(0, $initialSeasonals.size, ...$initialSeasonals.tile([$batchSize]).reshape([$batchSize, $seasonalPeriods]).transpose().reshape([$batchSize * $seasonalPeriods]).split($seasonalPeriods));
            }

            /// determin operands for trend detrend dampen
            let $trended = undefined; /// {"mul": np.multiply, "add": np.add, None: lambda l, b: l}
            let $detrend = undefined; /// {"mul": np.divide, "add": np.subtract, None: lambda l, b: 0}
            let $dampen = undefined; // {"mul": np.power, "add": np.multiply, None: lambda b, phi: 0}
            let $seasoned = undefined;

            if ($trendProcess === 0){
                $trended = ($level) => $level; /// identity 
                $detrend = () => $zeroScalar; 
                $dampen = () => $zeroScalar;
            }
            else if($trendProcess === 1){
                $trended = tf.add;
                $detrend = tf.sub;
                $dampen = tf.mul;
            }
            else if($trendProcess === 2){
                $trended = tf.mul;
                $detrend = tf.div;
                $dampen = tf.pow;
            }

            if($seasonalProcess === 0){
                $seasoned = ($i) => 
                    $i; /// identity 
            }
            else if($seasonalProcess === 1){
                $seasoned = tf.sub;
            }
            else if($seasonalProcess === 2){
                $seasoned = tf.div;
            }



            /// for i in range(1, nobs + 1):
            for(let i = 1; i < $observations + 1; i++){

                [
                    $levels[i],
                    $trends[i],
                    $seasonals[i + $seasonalPeriods - 1]
                ] = 
                /// cleanup intermediate tensors in every loop, export the level, trend, seasonal for this iterated period
                tf.tidy(() => {
                        
                    let $level = undefined;
                    let $trend = undefined;
                    let $seasonal = undefined;

                    /// shared computation chunks
                    const $levelTrendComponent = $trended($levels[i - 1], $dampen($trends[i - 1], $phiV)); /// alphac * trended(lvls[i - 1], dampen(b[i - 1], phi))
                    
                    /// compute level for iteration
                    let $levelSeasonalComponent = $yAlpha.gather([i - 1], 1);

                    if($seasonalProcess === 0){
                        /// pass
                    }
                    else if($seasonalProcess === 1){
                        $levelSeasonalComponent = $seasoned($levelSeasonalComponent, $seasonals[i - 1].mul($alphaV).transpose()); /// y_alpha[i - 1] - (alpha * s[i - 1]) + ... 
                    }
                    else if($seasonalProcess === 2){
                        $levelSeasonalComponent = $seasoned($levelSeasonalComponent, $seasonals[i - 1]); /// y_alpha[i - 1] / s[i - 1] + ... 
                    }

                    $level = $levelSeasonalComponent.add($levelTrendComponent.mul($alphaCV).transpose()).reshape([$batchSize]); /// y_alpha[i - 1] + ...

                    /// compute trend for iteration
                    if($trendProcess !== 0){

                        $trend = 
                            /// (beta * detrend(lvls[i], lvls[i - 1])) 
                            $detrend($level, $levels[i - 1]).mul($betaV)
                            /// + (betac * dampen(b[i - 1], phi)(
                            .add(
                                $dampen($trends[i - 1], $phiV).mul($betaCV)
                            )
                            .reshape([$batchSize]);

                    }

                    /// compute seasonals for iteration
                    if($seasonalProcess !== 0){

                        let $seasonalComponent = $yGamma.gather([i - 1], 1);

                        if($seasonalProcess === 1){
                            $seasonalComponent = $seasoned($seasonalComponent, $levelTrendComponent.mul($gammaV).transpose()); /// y_gamma[i - 1] - (gamma * trended(lvls[i - 1], dampen(b[i - 1], phi))) ... 
                        }
                        else if($seasonalProcess === 2){
                            $seasonalComponent = $seasoned($seasonalComponent, $levelTrendComponent);
                        }

                        $seasonal = $seasonalComponent.add($seasonals[i - 1].mul($gammaCV).transpose()).reshape([$batchSize]); // s[i + m - 1] = ... + 

                    }

                    return [
                        $level,
                        $trend,
                        $seasonal
                    ];

                });

            }

            /// copy trend / season before mutate / squeeze into one tensor ...
            const $lastObservedLevel = $levels[$observations];
            for(let i = 0; i < $predictions; i++){
                $levels[$observations + i + 1] = $lastObservedLevel.clone();
            }

            /// convert array of scalars to tensor vectors
            const $levelsVector = tf.stack($levels, 0);
            let $trendVector = undefined;
            
            if($trendProcess !== 0){
                $trendVector = tf.stack($trends, 0); // b[1 : nobs + 1].copy()
            }

            let $seasonsVector = undefined;
            
            if($seasonalProcess !== 0){
                $seasonsVector = tf.stack($seasonals, 0); // s[m : nobs + m].copy()
            }
            
            if($trendProcess !== 0){
                

                const $phiH = $damped ?
                    // np.cumsum(np.repeat(phi, h + 1) ** np.arange(1, h + 1 + 1))
                    $phis.tile([1, $predictions + 1]).pow(tf.range(1, $predictions + 2, 1, "int32")).cumsum(1) :
                    // np.arange(1, h + 1 + 1)
                    tf.range(1, $predictions + 2, 1, "int32"); /// just do column vect, will broadcast when dampen below

                /// dampen observations of $trendVector
                // b[:nobs] = dampen(b[:nobs], phi)
                const $dampendedTrendObservations = $dampen($trendVector, $phiV).slice([0], [$observations]);
                //  b[nobs:] = dampen(b[nobs], phi_h)
                const $dapendedTrendPredictions = $dampen($trendVector.gather([$observations]).transpose(), $phiH).transpose();

                /// stack dampended trends
                $trendVector = $dampendedTrendObservations.concat(
                    $dapendedTrendPredictions
                );

            }

            const $trendedLevelsVector = $trended($levelsVector, $trendVector);

            if($seasonalProcess !== 0){
                    
                /// get last $seasonalPeriods observations from seasonals
                /// and tile them over season vector end - 1 to $observations + $seasonalPeriods + $predictions + 1
                const $seasonalTile = $seasonsVector.slice([$observations - 1], [$seasonalPeriods]);
                const $seasonalPrediction = $seasonalTile.tile([Math.ceil(($predictions + 2) / $seasonalPeriods), 1]).slice([0], [$predictions + 2]);

                
                /// stack seasonals and tiled prediction seasons 
                $seasonsVector = 
                    $seasonsVector.slice([0], [$observations + $seasonalPeriods - 1])
                    .concat(
                        $seasonalPrediction
                    ).slice([0], [$observations + $predictions + 1]);


            }

            let $fittedVector = undefined;

            if($seasonalProcess === 0){

                $fittedVector = $trendedLevelsVector;
                
            }
            else if($seasonalProcess === 1){

                $fittedVector = $trendedLevelsVector.add($seasonsVector);

            }
            else if($seasonalProcess === 2){

                $fittedVector = $trendedLevelsVector.mul($seasonsVector);

            }

            /// BOX COX .... do box cox if needed on fitted data

            /// calculate redisduals
            const $observationsFitted =  
                $fittedVector.slice(
                    [0],
                    [$observations]
                );

            const $residuals = $endogenous.sub($observationsFitted.transpose());

            /// sum squared error
            const $sse = $residuals.pow($twoScalar).sum(1);
            /// 1/($actual.length-1)
            const $mse = $oneScalar.div($observationss.sub($oneScalar)).mul($sse);

            /// degrees of freedom
            const $degreesOfFreedom = tf.scalar($seasonalPeriods + (2 * !!$trendProcess) + 2 + (1 * !!$damped));

            /// other fitting statistics
            let $aic = undefined;
            let $aicc = undefined;
            let $bic = undefined;

            if($options.calculateInformationCriterion){

                const $degreesOfFreedomEff = $observationss.sub($degreesOfFreedom).sub($threeScalar);

                $aic = $sse.div($observationss).log().mul($observationss).add($degreesOfFreedom.mul($twoScalar));
                $aicc = $aic.add(($degreesOfFreedom.add($twoScalar).mul($twoScalar)).mul($degreesOfFreedom.add($threeScalar)).div($degreesOfFreedomEff));
                $bic = $observationss.mul($sse.div($observations).log()).add($degreesOfFreedom.mul($observationss.log()));

            }


            /// remove bias from fitted after calcing statistics
            /// if configured to do so
            if($removeBias){

                $fittedVector = $fittedVector.add($residuals.mean(1));

            }

            /// conform output to shape ($batchSize [, $observations [+ $predicions]])
            const $predictOutput = [
                $fittedVector.transpose(), /// ($batchSize, $observations + $predicions)
                $residuals, /// ($batchSize, $observations)
                $sse, /// ($batchSize)
                $mse, /// ($batchSize)
                $degreesOfFreedom, /// ()
                $aic, /// ($batchSize)
                $aicc, /// ($batchSize)
                $bic /// ($batchSize)
            ];

            if($options.recordOutputForSummary){
             
                this.$predictRecord.push($predictOutput);

            }

            return $predictOutput;

        });

    }

    dispose() {

        for(let i = 0; i < this.$predictRecord.length; i++){

            this.$predictRecord[i].forEach(
                $tensor => 
                    $tensor.dispose()
            );
            
        }

        for(let i = 0; i < this.$fitRecord.length; i++){

            this.$fitRecord[i].forEach(
                $tensor => 
                    $tensor.dispose()
            );
            
        }

    }

    summary () {

        if(!this.$predictRecord.length){

            return {};

        }

        const [
            ,
            , 
            $sse,
            $mse,
            $degreesOfFreedom, 
            $aic,
            $aicc,
            $bic
        ] = this.$predictRecord[this.$predictRecord.length - 1];

        return {
            startingParameters: {
                ...this.$parameters
            },
            fittedParameters: {
                alpha: this.$alpha,
                beta: this.$beta,
                gamma: this.$gamma
            },
            sse: $sse?.arraySync()[0],
            mse: $mse?.arraySync()[0],
            degreesOfFreedom: $degreesOfFreedom?.arraySync()[0],
            aic: $aic?.arraySync()[0],
            aicc: $aicc?.arraySync()[0],
            bic: $bic?.arraySync()[0]
        };

    }
    
    summaryText () {

        const $maxEpochRecords = 10;
        const $maxBatchRecords = 3;

        let $outputRope = "";

        if(this.$fitRecord.length){

            const $optimizerShortName = {
                0: "brute",
                1: "adam",
                2: "stochastic gradient descent",
                3: "adamax"
            }[this.$parameters.optimizer];

            $outputRope += `---------- OPTIMIZATION (${$optimizerShortName}) ----------\n`;
            $outputRope += "\n";
            $outputRope += `Learning Rate: ${this.$parameters.learningRate}\n`;
            $outputRope += `Batch Size: ${this.$parameters.optimizer === 0 ? this.$parameters.batchSize : "N/A"}\n`;
            $outputRope += `Epochs: ${this.$parameters.optimizer === 0 ? "N/A" : this.$parameters.epochs}\n`;
            $outputRope += `Regularizer Alpha: ${this.$parameters.regularizerAlpha === 0 ? "N/A" : this.$parameters.regularizerAlpha}\n`;
            $outputRope += "\n";
            
            /// output fitting history
            const $recordsToOutput = Math.min(this.$fitRecord.length, $maxEpochRecords);

            for(let i = 0; i < $recordsToOutput; i++){

                const [
                    $alpha,
                    $beta,
                    $gamma,
                    $phi,
                    $mse,
                    $aic,
                    $aicc,
                    $bic
                ] = this.$fitRecord[i].map(
                    $tensor =>
                        $tensor?.arraySync()
                );

                /// only show first 2 of the batch since there could be an extermely large number
                /// of fitted records
                const $batchRecordsToOutput = Math.min($alpha.length, $maxBatchRecords);

                for(let j = 0; j < $batchRecordsToOutput; j++){

                    $outputRope += `iteration (${i}, ${j})\tmse: ${$mse[j]}\taic: ${$aic[j]}\tbic: ${$bic[j]}\talpha: ${$alpha[j][0]}\tbeta: ${$beta?.[j][0]}\tgamma: ${$gamma?.[j][0]}\tphi: ${$phi?.[j][0]}\n`;

                }
                
                if($alpha.length > $maxBatchRecords){

                    $outputRope += `.... [${$alpha.length - $maxBatchRecords}] more in batch\n`;

                }
                
            }
                
            if(this.$fitRecord.length > $maxEpochRecords){

                $outputRope += `.... [${this.$fitRecord.length - $maxEpochRecords}] more epochs\n`;

            }

            
            $outputRope += "\n\n\n\n";

        }


        /// now output prediction summary

        
        const $lastPrediction = this.$predictRecord[this.$predictRecord.length - 1];

        if($lastPrediction){

            const [
                ,
                , 
                $sse,
                $mse,
                $degreesOfFreedom, 
                $aic,
                $aicc,
                $bic
            ] = $lastPrediction;

            let $shortNameRope = "";

            $shortNameRope += {
                0: "N",
                1: "A",
                2: "M"
            }[this.$parameters.trendProcess];

            if(this.$damped){

                $shortNameRope += "d";

            }

            $shortNameRope += {
                0: "N",
                1: "A",
                2: "M"
            }[this.$parameters.seasonalProcess];
            
            $outputRope += `---------- MODEL (${$shortNameRope})----------\n`;
            $outputRope += "\n";
            
            $outputRope += `Alpha: ${this.$alpha} | ${this.$parameters.alphaBounds}\n`;
            $outputRope += `Beta: ${this.$beta} | ${this.$parameters.betaBounds}\n`;
            $outputRope += `Gamma: ${this.$gamma} | ${this.$parameters.gammaBounds}\n`;
            $outputRope += `Phi: ${this.$phi} | ${this.$parameters.phiBounds}\n`;
            $outputRope += `Bias Removed: ${this.$parameters.removeBias ? "true" : "false"}\n`;
            $outputRope += `Initializer Type: ${this.$parameters.initialValuesGenerationType === 1 ? "estimate" : "heuristic"}`;
            $outputRope += "\n";
            $outputRope += `SSE: ${$sse.squeeze().arraySync()}\n`;
            $outputRope += `MSE: ${$mse.squeeze().arraySync()}\n`;
            $outputRope += `Degrees Of Freedom: ${$degreesOfFreedom.arraySync()}\n`;
            $outputRope += `AIC: ${$aic.squeeze().arraySync()}\n`;
            $outputRope += `AICC: ${$aicc.squeeze().arraySync()}\n`;
            $outputRope += `BIC: ${$bic.squeeze().arraySync()}\n`;

        }



        return $outputRope;

    }

}

/// export
globalThis.HoltWintersExponentialSmoothing = HoltWintersExponentialSmoothing;