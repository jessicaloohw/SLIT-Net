function [] = SLITNet_splitDatasets_withIndices(pathToDataset, trainIdxs, valIdxs, testIdxs, saveFolder, saveTag, deleteAfterSplit)
%
% SLIT-Net
% DOI: 10.1109/JBHI.2020.2983549
%

clearvars -except pathToDataset trainIdxs valIdxs testIdxs saveFolder saveTag deleteAfterSplit

% Load data:
allData = load(pathToDataset);
allData = allData.data;
nData = length(allData);
disp(['Dataset: ',num2str(nData),' images.']);

% Save training data:
if(~isempty(trainIdxs))
    data = allData(trainIdxs);
    save(fullfile(saveFolder,['train_data_',saveTag,'.mat']), '-v7.3', 'data');
    disp(['Training: ',num2str(length(data)),' images.']);
end

% Save validation data:
if(~isempty(valIdxs))
    data = allData(valIdxs);
    save(fullfile(saveFolder,['val_data_',saveTag,'.mat']), '-v7.3', 'data');
    disp(['Validation: ',num2str(length(data)),' images.']);
end

% Save testing data:
if(~isempty(testIdxs))
    data = allData(testIdxs);
    save(fullfile(saveFolder,['test_data_',saveTag,'.mat']), '-v7.3', 'data');
    disp(['Testing: ',num2str(length(data)),' images.']);
end

% Delete original data:
if(deleteAfterSplit)
   delete(pathToDataset);
   disp([pathToDataset,' deleted.']);
end

disp('Dataset split completed.');

end
