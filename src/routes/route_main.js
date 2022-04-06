import { Router } from 'express';
import * as dfd from 'danfojs-node';
import * as tf from '@tensorflow/tfjs';
import moment from 'moment';
import * as json2csv from 'json2csv';
import { writeFile } from 'fs/promises';

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
function cleanData(dataPath = '', viewOutput = false)
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
                ['LastName', 'Title', 'Sex', 'Embarked', 'AgeGroup', 'FareGroup'].forEach((columnName) => {
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

                // Remaining columns
                console.log('[INFO] Remaining column types');
                console.log(df.ctypes.index);

                // --- TENSORFLOW ---

                // Prepare the models
                let xTrain, yTrain;
                xTrain = df.iloc({ columns: [`1:`] });
                yTrain = df['Survived'];

                // Standardize the data with MinMaxScaler
                let scaler = new dfd.MinMaxScaler();
                scaler.fit(xTrain)
                xTrain = scaler.transform(xTrain)

                // Return data
                resolve({ xTrain, yTrain, passengerIDList, survivedList });
            })
            .catch((error) => reject(error.message));
    });
}
// endregion

mainRouter.get('/trainModel', (request, response) => {
    cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\train.csv", true)
        .then((result) => {
            const { xTrain, yTrain } = result;

            // Create the neural network with 4 layers
            const model = tf.sequential();
            model.add(tf.layers.dense({ inputShape: [10], units: 124, activation: 'relu', kernelInitializer: 'leCunNormal' }));
            model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
            model.summary();

            // Compile model
            model.compile({
                optimizer: "rmsprop",
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            });

            const startedAt = moment().unix();
            console.log('[STARTED] Model Training');

            model
                .fit(xTrain.tensor, yTrain.tensor, {
                    batchSize: 32,
                    epochs: 35,
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
                    cleanData("D:\\Projects\\OU\\TitanicML\\src\\data\\test.csv", false)
                        .then((testModel) => {
                            // Predict model
                            let predictResult = model.predict(testModel.xTrain.tensor).dataSync();

                            // Map into submission data format
                            let submissionArr = testModel.passengerIDList.map((id, index) => ({ 'PassengerId': id, 'Survived': Math.round(predictResult[index]) }));

                            // Convert JSON to CSV
                            const csv = json2csv.parse(submissionArr);

                            // Write to file
                            writeFile(`./predictedResult.csv`, csv)
                                .then(() => Success(response, 'Successfully trained model, took ' + moment().unix() - startedAt + 's', 200))
                                .catch((writeError) => Abort(response, 'Write Error', 500, writeError.message));
                        });
                })
                .catch((error) => console.log('[ERROR] Model Training', error.message));

            // ------
        })
        .catch((error) => Abort(response, 'Failed to read TRAIN file', 500, error));
});

export { mainRouter };
