# holt_winters_exp_smooth_tf
Partial port of statsmodels Holt Winters exponential smoothing library using tensorflow for matrix operations and optimization


**Requirements**

1. tensorflow must be imported, ready, and available via global scope variable "tf"

**Example**

```javascript

/// load tensorflow (can be loaded any way as long as tf is assigned to globalThis or window)
globalThis.tf = require('@tensorflow/tfjs');

/// put in prod mode
tf.enableProdMode();

/// run async context
(async () => {

    const $endogenousData = [ 1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1 ];

    const $cancellable = {
        checkCancelled: () => {

            throw new Error({ cancelled: true });

        }
    };

    let $model = undefined;

    try {
        
        $model = new HoltWintersExponentialSmoothing(
            $endogenousData,
            {
                
                /*
                    optimizer (only applicable if fit() is called):
                    0 -> brute force fit.
                    1 -> adam optimizer
                    2 -> stochastic gradient descent optimizer
                    3 -> adamax optimizer
                */
                optimizer: 0, 
                /// learning rate. For brute force optmizer will predict over every parameter in parameter space, over respective parameter bounds, incrementing at the learning rate. For other optimizers is the learning rate hyper parameter of the respective optimizer
                learningRate: 0.1,
                /// batch size is only applicable to brute force optimizer mode. Is the number of predictions to make at once over the parameter space. A higher batch size consumes more memory but is exponentially faster
                batchSize: 2048,
                /// epochs is only applicable in non brute force optimizers. This is the number of iterations of the optimizer to run before quiting
                epochs: 20,
                /// A hyper parameter used to helop clamp the model parameters to their bounds fro non brute force fit optimizers. Set to 0 to disable this
                regularizerAlpha: 0.5,

                /// number of periods of a season. Must be > 0 if seasonalProcess !== 0 
                seasonalPeriods: 0,
                /// the seasonal process. 0: None, 1: Additive, 2: Multiplicative
                seasonalProcess: 1,
                /// the trend process. 0: None, 1: Additive, 2: Multiplicative
                trendProcess: 1,
                /// the dampen process. 0: None, 1: Auto (mirrors trend process)
                dampenProcess: 1, /// 0 | 1

                /// the initial values generation type (not used ATM)
                initialValuesGenerationType: 1,

                /// the model values used for predict() call if fit() is not called priorly. If fit() is called then these are the starting values of the non brute optimizers
                alpha: 0, /// level smoothing factor
                beta: 0, /// trend smoothing factor
                gamma: 0, /// seasonal smoothing factor
                phi: 1, /// dampening smoothing factor

                /// the boundaries of the model parameters
                alphaBounds: [0, 1],
                betaBounds: [0, 1],
                gammaBounds: [0, 1],
                phiBounds: [0, 1],

                /// remove mean of fitted observation residuals from predicted output
                removeBias: false
            },
            /// (optional) you can pass an object with property called "checkCancelled" that will be called periodicly. You can take the opportunity to throw an error to quit execution
            $cancellable
        );

        /// fit the model if you want the model parameters to be optimized
        /// automaticly using passed configuration in constructor. Otherwise skip
        /// to predict
        await $model.fit();

        /// b/c of the way predict is designed to batch predictions internally will returns array of tensors
        const $predictionOutput = $model.predict(
            25 /// number of periods to predict
        ); 

        const [
            $fitted, /// Array<Array<Number>>
            $residuals, /// Array<Array<Number>>
            $sse, /// Array<Number>
            $mse, /// Array<Number>
            $degreesOfFreedom,  /// Number
            $aic, /// Array<Number>
            $aicc, /// Array<Number>
            $bic /// Array<Number>
        ] =
        await Promise.all(
            $predictionOutput.map(
                $output => $output.array()
            )
        );

        /// summary output
        console.log($model.summary());
        /// -> { startingParameters: ..., fittedParameters: ..., sse: ..., mse: ..., degreesOfFreedom: ..., aic: ..., aicc: ..., bic: ... }

        
        /// nice formatted text output of the modeling / prediction process
        console.log($model.summaryText());

    }
    finally {

        /// make sure to dispose the model or else will leak 
        /// tensors
        $model?.dispose();

    }

})();


```