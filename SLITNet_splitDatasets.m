%
% Split datasets into7-fold (7 x 19 = 133):
%

% Common settings:
pathToDataset = 'Datasets/annotations.mat';
saveFolder = 'Datasets';
pathToIdxs = 'Datasets/idxs.mat';
idxsToExclude = [];
deleteAfterSplit = false;

load(pathToIdxs); % Loads variables k1, k2, k3, k4, k5, k6, k7

for k = 1:7
    
    switch k
        case 1
            trainIdxs = [k2, k3, k4, k5, k6];
            valIdxs = k7;
            testIdxs = k1;
            saveTag = 'K1';
        case 2
            trainIdxs = [k3, k4, k5, k6, k7];
            valIdxs = k1;
            testIdxs = k2;
            saveTag = 'K2';
        case 3
            trainIdxs = [k4, k5, k6, k7, k1];
            valIdxs = k2;
            testIdxs = k3;
            saveTag = 'K3';
        case 4
            trainIdxs = [k5, k6, k7, k1, k2];
            valIdxs = k3;
            testIdxs = k4;
            saveTag = 'K4';
        case 5
            trainIdxs = [k6, k7, k1, k2, k3];
            valIdxs = k4;
            testIdxs = k5;
            saveTag = 'K5';
        case 6
            trainIdxs = [k7, k1, k2, k3, k4];
            valIdxs = k5;
            testIdxs = k6;
            saveTag = 'K6';
        case 7
            trainIdxs = [k1, k2, k3, k4, k5];
            valIdxs = k6;
            testIdxs = k7;
            saveTag = 'K7';
    end
    
    % Add one because index starts from 0:
    trainIdxs = sort(trainIdxs) + 1;
    valIdxs = sort(valIdxs) + 1;
    testIdxs = sort(testIdxs) + 1;
    
    SLITNet_splitDatasets_withIndices(pathToDataset, trainIdxs, valIdxs, testIdxs, saveFolder, saveTag, deleteAfterSplit);
end
