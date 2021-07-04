const loadModel = async () => {
    const url =
        'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json';
    const model = await tf.loadLayersModel(url);
    return model;
};

const loadMetadata = async () => {
    const url =
        'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json';
    const metadata = await fetch(url);
    return metadata.json();
};

const padSequences = (sequences, metadata) => {
    return sequences.map((seq) => {
        if (seq.length > metadata.max_len) {
            seq.splice(0, seq.length - metadata.max_len);
        }
        if (seq.length < metadata.max_len) {
            const pad = [];
            for (let i = 0; i < metadata.max_len - seq.length; i++) {
                pad.push(0);
            }
            seq = pad.concat(seq);
        }
        return seq;
    });
};

const predict = (text, model, metadata) => {
    const trimmed = text
        .trim()
        .toLowerCase()
        .replace(/(\.|\,|\!|\?)/g, '')
        .split(' ');

    const sequence = trimmed.map((word) => {
        const wordIndex = metadata.word_index[word];
        if (typeof wordIndex === 'undefined') {
            return 2;
        }
        return wordIndex + metadata.index_from;
    });

    const paddedSequence = padSequences([sequence], metadata);
    const input = tf.tensor2d(paddedSequence, [1, metadata.max_len]);

    const predictOut = model.predict(input);

    const score = predictOut.dataSync()[0];
    return score;
};

const getSentiment = (score) => {
    if (score > 0.66) {
        return `Score of ${score} is Positive`;
    } else if (score > 0.4) {
        return `Score of ${score} is Neutral`;
    } else {
        return `Score of ${score} is Negative`;
    }
};

const run = async (text) => {
    const model = await loadModel();
    const metadata = await loadMetadata();

    let sum = 0;

    text.forEach((prediction) => {
        const perc = predict(prediction, model, metadata);
        sum += parseFloat(perc, 10);
    });

    console.log(getSentiment(sum / text.length));
};

window.onload = () => {
    const inputText = document.getElementsByTagName('input')[0];
    const button = document.getElementsByTagName('button')[0];

    button.onclick = () => {
        run([inputText.value]);
    };
};
