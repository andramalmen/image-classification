'use strict';

const app = async () => {
    const webcamElement = document.getElementsByClassName('webcam')[0];
    const buttons = document.getElementsByTagName('button');
    const predictButton = document.getElementsByClassName('predict')[0];
    const classes = ['up', 'down', 'left', 'right'];
    const predictionParagraph =
        document.getElementsByClassName('prediction')[0];

    const classifier = knnClassifier.create();
    const net = await mobilenet.load();
    const webcam = await tf.data.webcam(webcamElement);
    // console.log(tf.data.webcam());

    const addExample = async (classId) => {
        const img = await webcam.capture();
        const activation = net.infer(img, 'conv_preds');
        classifier.addExample(activation, classId);
        img.dispose();
    };

    for (let i = 0; i < buttons.length; i++) {
        if (buttons[i] !== predictButton) {
            let index = i;
            buttons[i].onclick = () => addExample(index);
        }
    }

    const runPredictions = async () => {
        while (true) {
            if (classifier.getNumClasses() > 0) {
                const img = await webcam.capture();
                const activation = net.infer(img, 'conv_preds');
                const result = await classifier.predictClass(activation);

                predictionParagraph.innerHTML = `
                    prediction: ${classes[result.label]},
                    probability: ${result.confidences[result.label]}
                `;

                img.dispose();
            }
            await tf.nextFrame();
        }
    };

    predictButton.onclick = () => runPredictions();
};

app();
