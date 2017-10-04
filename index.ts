import * as Jimp from "jimp";
import { Neuron, Network, Trainer, Architect } from "synaptic";

async function getImgData(fileName: string) {
  let image = await Jimp.read(fileName);
  let inputSet: any = [];
  image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
    let red = image.bitmap.data[idx + 0];
    let green = image.bitmap.data[idx + 1];
    let blue = image.bitmap.data[idx + 2];
    let alpha = image.bitmap.data[idx + 3];

    inputSet.push([red, green, blue, alpha]);
  });
  return inputSet;
}

async function main() {
  // 4 inputs, 5 node hidden layer, 5 node hidden layer and 4 outputs
  try {
    const perceptron = new Architect.Perceptron(4, 5, 5, 4);
    const trainer = new Trainer(perceptron);
    const trainingSet: any = [];

    let inputs: any = await getImgData('input_image_train.jpg');
    let outputs: any = await getImgData('output_image_train.jpg');

    for (let i = 0; i < inputs.length; i++) {
      trainingSet.push({
        input: inputs[i].map((val: any) => val / 255),
        output: outputs[i].map((val: any) => val / 255)
      });
    }

    trainer.train(trainingSet, {
      rate: .1,
      iterations: 10,
      error: .005,
      shuffle: true,
      log: 10,
      cost: Trainer.cost.CROSS_ENTROPY
    });

    let image = await Jimp.read('test.jpg');
    image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
      let red = image.bitmap.data[idx + 0];
      let green = image.bitmap.data[idx + 1];
      let blue = image.bitmap.data[idx + 2];
      let alpha = image.bitmap.data[idx + 3];

      let out = perceptron.activate([red / 255, green / 255, blue / 255, alpha / 255]);

      image.bitmap.data[idx + 0] = Math.round(out[0] * 255);
      image.bitmap.data[idx + 1] = Math.round(out[1] * 255);
      image.bitmap.data[idx + 2] = Math.round(out[2] * 255);
    });
    console.log('Writing output to file: out.jpg');
    image.write('out.jpg');

  }
  catch (err) {
    console.error(err);
  }
}
main();
