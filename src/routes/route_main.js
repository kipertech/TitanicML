import { Router } from 'express';
import * as dfd from 'danfojs-node';
import * as tf from '@tensorflow/tfjs';
import moment from 'moment';
import * as json2csv from 'json2csv';
import path from 'path';
import { writeFile } from 'fs/promises';

import { RandomForestClassifier, RandomForestRegression } from 'ml-random-forest';
import { RandomForestClassifier as rfClassifier, RandomForestRegressor as rfRegression } from 'random-forest';

import LogisticRegression from 'ml-logistic-regression';
import { Matrix } from 'ml-matrix';

import { DecisionTreeClassifier, DecisionTreeRegression } from 'ml-cart';

import { Abort, Success } from '../utils';

const mainRouter = Router();

// region Function - Print Data
function printData(df, viewOutput = false, message = '')
{
    if (viewOutput)
    {
        console.log('[INFO] ' + message);
        df.head(10).print();
    }
}
// endregion

// region Function - Clean Data
function cleanData(dataPath = '', viewOutput = false, standardize = true)
{
    return new Promise((resolve, reject) => {
        dfd
            .readCSV(dataPath)
            .then((result) => {
                let df = result;

                // Save the list of passenger ID
                const passengerIDList = df['PassengerId'].values, survivedList = df['Survived']?.values || [];

                // View Data
                printData(df, viewOutput, 'Original Data');

                // Since the fare price includes family ticket prices as well,
                // we need to divide the fare price the passenger to their (number of siblings/spouse + number of parents/children)
                let fareList = df['Fare'].values,
                    sibSpList = df['SibSp'].values,
                    parChList = df['Parch'].values;

                // Add new column for Family Size
                let familySize = passengerIDList.map((id, index) => Number(sibSpList[index] || 0) + Number(parChList[index] || 0) + 1);
                df = df.addColumn('FamilySize', familySize);

                // Calculate new fare price
                let newFareList = fareList.map((fare, index) => {
                    return(
                        Number(
                            (
                                Number(fare || 0)
                                /
                                ((Number(sibSpList[index] || 0) + Number(parChList[index] || 0)) /* Total number of other family members */ + 1 /* The passenger themselves */)
                            )
                                .toFixed(2)
                        )
                    )
                });

                df = df.drop({ columns: ['Fare'] });
                df = df.addColumn('Fare', newFareList);

                // Fill missing values for "Age"
                const ageMedian = df['Age'].median();
                df = df.fillNa(ageMedian, { columns: ['Age'] });

                // Fill missing values for "Embarked"
                df = df.fillNa('S', { columns: ['Embarked'] });

                // Create groups of Age
                // 0-12 -> Children; 13-20 -> Teenager; 21-40 -> Adult; 41+ -> Elder
                let ageGroup = df['Age'].values.map((age) => {
                    if (age <= 12) return 'Children';
                    else if (age > 12 && age <= 20) return 'Teenager';
                    else if (age > 20 && age <= 40) return 'Adult';
                    else return 'Elder';
                });

                df = df.drop({ columns: ['Age'] });
                df = df.addColumn('AgeGroup', ageGroup);

                // Create groups of Fare
                // 0-8 -> Low; 8.01-15 -> Median; 15.01-31 -> Average; 31.01+ -> High
                let fareGroup = df['Fare'].values.map((fare) => {
                    if (fare <= 8) return 'Low';
                    else if (fare > 8 && fare <= 15) return 'Median';
                    else if (fare > 15 && fare <= 31) return 'Average';
                    else return 'High';
                });

                df = df.drop({ columns: ['Fare'] });
                df = df.addColumn('FareGroup', fareGroup);

                // Extract name components
                let rawNames = df['Name'].values,
                    firstNames = rawNames.map((item) => item.split(', ')[0]),
                    lastNames = rawNames.map((item) => item.split('. ')[1]?.split(' ')[0]),
                    titles = rawNames.map((item) => item.split(', ')[1]?.split('. ')[0]);

                // Add new columns to dataframe
                df = df.addColumn('Title', titles);
                df = df.addColumn('FirstName', firstNames);
                df = df.addColumn('LastName', lastNames);

                // Remove the original column "Name"
                df = df.drop({ columns: ['Name'] });

                // View data
                printData(df, viewOutput, 'After name components extraction');

                // Standardize Titles
                let manList = ["Capt", "Don", "Major", "Col", "Rev", "Dr", "Sir", "Mr", "Jonkheer"],
                    womanList = ["Dona", "the Countess", "Mme", "Mlle", "Ms", "Miss", "Lady", "Mrs"];

                let newTitleData = df['Title'].values.map((title) => {
                    if (manList.includes(title))
                    {
                        return("man");
                    }
                    else if (womanList.includes(title))
                    {
                        return("woman");
                    }
                    else return("boy");
                });

                // Put the data back into dataframe and remove the old Title column
                df = df.drop({ columns: ['Title'] });
                df = df.addColumn('Title', newTitleData);

                // View data
                printData(df, viewOutput, 'After Title parsing');

                // See the gender count
                printData(df, viewOutput, 'Gender Count');

                // Encode the columns
                ['LastName', 'Title', 'Sex', 'Embarked', 'AgeGroup', 'FareGroup', 'FamilySize'].forEach((columnName) => {
                    let encoder = new dfd.LabelEncoder(),
                        dataSeries = new dfd.Series(df[columnName].values);

                    encoder.fit(dataSeries);

                    // Replace the entire column
                    df = df.drop({ columns: [columnName] });
                    df = df.addColumn(columnName, encoder.transform(dataSeries.values));
                });

                // View Data
                printData(df, viewOutput, 'After Title and Sex encoding');

                // Percentage of missing Cabin information
                let nullCount = Number(df['Cabin'].valueCounts().values[0]),
                    totalRowCount = df['PassengerId'].values.length;

                if (viewOutput) console.log('[INFO] Missing Cabin information: ' + (nullCount / totalRowCount * 100).toFixed(2) + '%');

                // Dropping unnecessary columns
                df = df.drop({ columns: ['Cabin', 'PassengerId', 'Ticket', 'FirstName'] });
                printData(df, viewOutput, 'After dropping PassengerId, Cabin, Ticket, and FirstName');

                // Remaining empty values
                console.log('[INFO] Missing values check');
                df.isNa().sum().print();

                // Remaining columns
                console.log('[INFO] Remaining column types');
                console.log(df.ctypes.index);

                // --- TENSORFLOW ---

                // Prepare the models
                let xTrain, yTrain;
                xTrain = df.iloc({ columns: [`1:`] });
                yTrain = df['Survived'];

                // Standardize the data with MinMaxScaler
                if (standardize)
                {
                    let scaler = new dfd.MinMaxScaler();
                    scaler.fit(xTrain)
                    xTrain = scaler.transform(xTrain)
                }

                // Number of total final column count (for Tensorflow shape count)
                const columnCount = df.ctypes.index.length;

                // Return data
                resolve({ xTrain, yTrain, passengerIDList, survivedList, columnCount });
            })
            .catch((error) => reject(error.message));
    });
}
// endregion

// region Tensorflow
mainRouter.get('/trainModel', (request, response) => {
    let trainData = path.join(__dirname, '../data/train.csv'),
        testData = path.join(__dirname, '../data/test.csv');

    cleanData(trainData, true)
        .then((result) => {
            const { xTrain, yTrain, columnCount } = result;

            // Create the neural network with 4 layers
            const model = tf.sequential();
            model.add(tf.layers.dense({ inputShape: [columnCount - 1], units: 124, activation: 'relu', kernelInitializer: 'leCunNormal' }));
            model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 1, activation: 'sigmoid', kernelInitializer: 'leCunNormal' }));
            model.summary();

            // Compile model
            model.compile({
                optimizer: 'adam',
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            });

            const startedAt = moment().unix();
            console.log('[STARTED] Model Training');

            model
                .fit(xTrain.tensor, yTrain.tensor, {
                    batchSize: 32,
                    epochs: 50,
                    validationSplit: 0.2,
                    callbacks: {
                        onEpochEnd: (epoch, logs) => {
                            let trainAcc = (logs.acc * 100).toFixed(2),
                                valAcc = (logs.val_acc * 100).toFixed(2);

                            console.log(`EPOCH (${epoch + 1}): Train Accuracy: ${trainAcc}, Val Accuracy:  ${valAcc}\n`);
                        }
                    }
                })
                .then(() => {
                    // Create models for TEST data
                    cleanData(testData, false)
                        .then((testModel) => {
                            // Predict model
                            let predictResult = model.predict(testModel.xTrain.tensor).dataSync();

                            // Map into submission data format
                            let submissionArr = testModel.passengerIDList.map((id, index) => ({ 'PassengerId': id, 'Survived': Math.round(predictResult[index]) }));

                            // Write to file
                            writeFile(`./predictedResult.csv`, json2csv.parse(submissionArr))
                                .then(() => Success(response, 'Successfully trained model, took ' + moment().unix() - startedAt + 's', 200))
                                .catch((writeError) => Abort(response, 'Write Error', 500, writeError.message));
                        });
                })
                .catch((error) => console.log('[ERROR] Model Training', error.message));

            // ------
        })
        .catch((error) => Abort(response, 'Failed to read TRAIN file', 500, error));
});
// endregion

// region Random Forest - Classifier
mainRouter.get('/randomForestClassifier', (request, response) => {
    const startedAt = moment().unix();

    cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\train.csv", true, false)
        .then((result) => {
            const { xTrain, yTrain, columnCount } = result;

            const options = {
                seed: 3,
                maxFeatures: 0.8,
                replacement: true,
                nEstimators: 25
            };

            const classifier = new RandomForestClassifier(options);
            classifier.train(xTrain.values, yTrain.values);

            // Create models for TEST data
            cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\test.csv", false, false)
                .then((testModel) => {
                    // Predict model
                    const predictedResult = classifier.predict(testModel.xTrain.values);

                    // Map into submission data format
                    let submissionArr = testModel.passengerIDList.map((id, index) => ({ 'PassengerId': id, 'Survived': Math.round(predictedResult[index]) }));

                    // Write to file
                    writeFile(`./predictedResult.csv`, json2csv.parse(submissionArr))
                        .then(() => Success(response, 'Successfully trained model, took ' + moment().unix() - startedAt + 's', 200))
                        .catch((writeError) => Abort(response, 'Write Error', 500, writeError.message));
                });
        })
        .catch((error) => Abort(response, 'Failed to read TRAIN file', 500, error.message));
});
// endregion

// region Random Forest - Regression
mainRouter.get('/randomForestRegression', (request, response) => {
    const startedAt = moment().unix();

    cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\train.csv", true, true)
        .then((result) => {
            const { xTrain, yTrain } = result;

            const options = {
                seed: 3,
                maxFeatures: 2,
                replacement: false,
                nEstimators: 200
            };

            const regression = new RandomForestRegression(options);
            regression.train(xTrain.values, yTrain.values);

            // Create models for TEST data
            cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\test.csv", false, true)
                .then((testModel) => {
                    // Predict model
                    const predictedResult = regression.predict(testModel.xTrain.values);

                    // Map into submission data format
                    let submissionArr = testModel.passengerIDList.map((id, index) => ({ 'PassengerId': id, 'Survived': Math.round(predictedResult[index]) }));

                    // Write to file
                    writeFile(`./predictedResult_RandomForest_Regression.csv`, json2csv.parse(submissionArr))
                        .then(() => Success(response, 'Successfully trained model, took ' + moment().unix() - startedAt + 's', 200))
                        .catch((writeError) => Abort(response, 'Write Error', 500, writeError.message));
                });
        })
        .catch((error) => Abort(response, 'Failed to read TRAIN file', 500, error.message));
});
// endregion

// region Random Forest - Classifier 2
mainRouter.get('/randomForestClassifier2', (request, response) => {
    const startedAt = moment().unix();

    cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\train.csv", true, false)
        .then((result) => {
            const { xTrain, yTrain, columnCount } = result;

            const rf = new rfClassifier({
                nEstimators: 200,
                maxDepth: 20,
                maxFeatures: 'auto',
                minSamplesLeaf: 15,
                minInfoGain: 0
            });

            rf.train(xTrain.values, yTrain.values);

            // Create models for TEST data
            cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\test.csv", false, false)
                .then((testModel) => {
                    // Predict model
                    const predictedResult = rf.predict(testModel.xTrain.values);

                    // Map into submission data format
                    let submissionArr = testModel.passengerIDList.map((id, index) => ({ 'PassengerId': id, 'Survived': Math.round(predictedResult[index]) }));

                    // Write to file
                    writeFile(`./predictedResult_RandomForest_Classifier2.csv`, json2csv.parse(submissionArr))
                        .then(() => Success(response, 'Successfully trained model, took ' + (moment().unix() - startedAt).toString() + 's', 200))
                        .catch((writeError) => Abort(response, 'Write Error', 500, writeError.message));
                });
        })
        .catch((error) => Abort(response, 'Failed to read TRAIN file', 500, error.message));
});
// endregion

// region Random Forest - Regression 2
mainRouter.get('/randomForestRegression2', (request, response) => {
    const startedAt = moment().unix();

    cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\train.csv", true, true)
        .then((result) => {
            const { xTrain, yTrain } = result;

            const rf = new rfRegression({
                nEstimators: 100,
                maxDepth: 10,
                maxFeatures: 'auto',
                minSamplesLeaf: 5,
                minInfoGain: 0
            });

            rf.train(xTrain.values, yTrain.values);

            // Create models for TEST data
            cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\test.csv", false, true)
                .then((testModel) => {
                    // Predict model
                    const predictedResult = rf.predict(testModel.xTrain.values);

                    // Map into submission data format
                    let submissionArr = testModel.passengerIDList.map((id, index) => ({ 'PassengerId': id, 'Survived': Math.round(predictedResult[index]) }));

                    // Write to file
                    writeFile(`./predictedResult_RandomForest_Regression2.csv`, json2csv.parse(submissionArr))
                        .then(() => Success(response, 'Successfully trained model, took ' + (moment().unix() - startedAt).toString() + 's', 200))
                        .catch((writeError) => Abort(response, 'Write Error', 500, writeError.message));
                });
        })
        .catch((error) => Abort(response, 'Failed to read TRAIN file', 500, error.message));
});
// endregion

// region Logistic Regression
mainRouter.get('/logisticRegression', (request, response) => {
    const startedAt = moment().unix();

    cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\train.csv", true, true)
        .then((result) => {
            const { xTrain, yTrain } = result;

            const logReg = new LogisticRegression({ numSteps: 1000, learningRate: 5e-3 });
            logReg.train(new Matrix(xTrain.values), Matrix.columnVector(yTrain.values));

            // Create models for TEST data
            cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\test.csv", false, true)
                .then((testModel) => {
                    // Predict model
                    const predictedResult = logReg.predict(new Matrix(testModel.xTrain.values));

                    // Map into submission data format
                    let submissionArr = testModel.passengerIDList.map((id, index) => ({ 'PassengerId': id, 'Survived': Math.round(predictedResult[index]) }));

                    // Write to file
                    writeFile(`./predictedResult_logisticRegression.csv`, json2csv.parse(submissionArr))
                        .then(() => Success(response, 'Successfully trained model, took ' + (moment().unix() - startedAt).toString() + 's', 200))
                        .catch((writeError) => Abort(response, 'Write Error', 500, writeError.message));
                });
        })
        .catch((error) => Abort(response, 'Failed to read TRAIN file', 500, error.message));
});
// endregion

// region Decision Tree - CART
mainRouter.get('/decisionTree', (request, response) => {
    const startedAt = moment().unix();

    cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\train.csv", true, true)
        .then((result) => {
            const { xTrain, yTrain } = result;

            const classifier = new DecisionTreeRegression();
            classifier.train(xTrain.values, yTrain.values);

            // Create models for TEST data
            cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\test.csv", false, true)
                .then((testModel) => {
                    // Predict model
                    const predictedResult = classifier.predict(testModel.xTrain.values);

                    // Map into submission data format
                    let submissionArr = testModel.passengerIDList.map((id, index) => ({ 'PassengerId': id, 'Survived': Math.round(predictedResult[index]) }));

                    // Write to file
                    writeFile(`./predictedResult_decisionTree.csv`, json2csv.parse(submissionArr))
                        .then(() => Success(response, 'Successfully trained model, took ' + (moment().unix() - startedAt).toString() + 's', 200))
                        .catch((writeError) => Abort(response, 'Write Error', 500, writeError.message));
                });
        })
        .catch((error) => Abort(response, 'Failed to read TRAIN file', 500, error.message));
});
// endregion

export { mainRouter };
