pkg load io

outputCell = [];

tol = 1e-10;

## Algorithms and SVs chosen from training results
param = [1 2 22 24; 5 2 16 26 ; 2 2 22 26; 15 2 18 20; 6 2 8 18; 20 2 16 20]
for row = 1:6
algorithm = param(row,1)
for col = 1:3
singularvectors = param(row, 1+col)

## For tagging the results table
alg = {"EDEIM", "LDEIM", "INF_LDEIM", "ONE_LDEIM", "DEIMQR", "QDEIM_EDEIM", "LEVERAGE_EDEIM", "QR_EDEIM", "OneLEVERAGE_EDEIM", "InfLEVERAGE_EDEIM", "QR","DEIM","QDEIM","DEIMCSDEIM","QDEIMCSDEIM","QRCSQR","DEIMCSQR","QDEIMCSQR","QRCSDEIM","LeverageScore"};

alg(algorithm)

## Patient file numbers in the training dataset
q = [100;103;105;111;113;117;121;123;200;202;210;212;213;214;219;221;222;228;231;232;233;234];

numberOfPatients = max(size(q));
binarycount = zeros(1,17);

##Classes from the README. The number sign is for anything that doesn't fit.
key = {"N", "A", "V", "Q", "/", "f", "F", "j", "L", "a", "J", "R", "!", "E", "s", "e", "#"};

output = zeros(1,17);
subsetsum = zeros(1,17);
fullsetsum = zeros(1,17);
totalSelections = 0;
totalHeartbeats = 0;

##Selection limit is double the number of classes in training or testing sets
selectionsLimit = 28;
for patientNumber = 1:numberOfPatients
    filename = [int2str(q(patientNumber,:)) "m_filtered_data_matrix.mat"];
    load(["data/" filename]);
    heartbeats = max(size(info.data_matrix1));
    B = info.data_matrix1;
    [U,S,V] = svd(full(B),'econ');
    morphcountsfullset = condense_list(info.annotations);
    morphcountsfullset(morphcountsfullset ~= 0) = 1;
    fullsetsum = fullsetsum + morphcountsfullset;
    if algorithm == 1 #EDEIM
        [p] = NewEDeim(U(:,1:singularvectors),'mem_coh',1e-4,selectionsLimit);
    elseif algorithm == 2 #LDEIM
        [p] = LDEIMv4(U(:,1:singularvectors),selectionsLimit);
    elseif algorithm == 3 #Inf_LDEIM
        [p] = ModifiedLDEIM(U(:,1:singularvectors),selectionsLimit,inf);
    elseif algorithm == 4 #One_LDEIM
        [p] = ModifiedLDEIM(U(:,1:singularvectors),selectionsLimit,1);
    elseif (algorithm == 5) #DEIMQR
        [p] = DEIMQR(U(:,1:singularvectors),selectionsLimit);
    elseif algorithm == 6 #QDEIM_EDEIM
        [p] = QDEIMandRestartedDEIMv2(U(:,1:singularvectors),'mem_coh',tol,selectionsLimit);
    elseif algorithm == 7 #LEV_EDEIM
        [p] = LeverageandRestartedDEIMv2(U(:,1:singularvectors),'mem_coh',tol,selectionsLimit,2);
    elseif algorithm == 8 %QR EDEIM
        [~,~,p] = qr(B', 'vector');
        limit = min([selectionsLimit, singularvectors]);
        p = p(1:limit)';
        p = RestartedDEIM(p,U(:,1:singularvectors),'mem_coh',tol,selectionsLimit);
    elseif algorithm == 9 #One_LEV_EDEIM
        [p] = LeverageandRestartedDEIM(U(:,1:singularvectors),'mem_coh',tol,selectionsLimit,1);
    elseif algorithm == 10  #INF_LEV_EDEIM
        [p] = LeverageandRestartedDEIM(U(:,1:singularvectors),'mem_coh',tol,selectionsLimit,inf);
    elseif algorithm == 11 %QR
        [~,~,p] = qr(B', 'vector');
        p = p(1:selectionsLimit)';
    elseif algorithm == 12 %DEIM
        [p] = NewDeim(U(:,1:singularvectors),selectionsLimit);
    elseif algorithm == 13 %QDEIM
        [p,~] = Newq_deim(U(:,1:singularvectors),selectionsLimit);
    elseif algorithm == 14 %DEIMCSDEIM
        [p] = NewDeim(U(:,1:singularvectors),selectionsLimit);
        [p] = CSDEIM(U(:,1:singularvectors),p,selectionsLimit-singularvectors);
    elseif algorithm == 15 %QDEIMCSDEIM
        [p,~] = Newq_deim(U(:,1:singularvectors),selectionsLimit);
        [p] = CSDEIM(U(:,1:singularvectors),p',selectionsLimit-singularvectors);
    elseif algorithm == 16 %QRCSQR
        [~,~,p] = qr(B', 'vector');
        limit = min([selectionsLimit, singularvectors]);
        p = p(1:limit);
        [p] = CSQR(U(:,1:singularvectors),p,selectionsLimit-singularvectors);
    elseif algorithm == 17 %DEIMCSQR
        [p] = NewDeim(U(:,1:singularvectors),selectionsLimit);
        p = p';
        [p] = CSQR(U(:,1:singularvectors),p,selectionsLimit-singularvectors);
    elseif algorithm == 18 %QDEIMCSQR
        [p,~] = Newq_deim(U(:,1:singularvectors),selectionsLimit);
        [p] = CSQR(U(:,1:singularvectors),p,selectionsLimit-singularvectors);
    elseif algorithm == 19 %QRCSDEIM
        [~,~,p] = qr(B', 'vector');
        limit = min([selectionsLimit, singularvectors]);
        p = p(1:limit)';
        [p] = CSDEIM(U(:,1:singularvectors),p,selectionsLimit-singularvectors);
    elseif algorithm == 20 %LeverageScore
        [p] = LeverageScoreOS(U(:,1:singularvectors),selectionsLimit,2);
    end
    morphcountssubset = condense_list(info.annotations(p));
    morphcountssubset(morphcountssubset ~= 0) = 1;
    subsetsum = subsetsum + morphcountssubset;
    totalSelections = totalSelections + max(size(p));
    totalHeartbeats = totalHeartbeats + heartbeats;
  end


selectionsAvg = totalSelections / numberOfPatients;
weightedTruncationAvg = totalSelections / totalHeartbeats;


OutputMatrix = [subsetsum' fullsetsum' repelems(singularvectors,[1;17])' repelems(selectionsLimit,[1;17])' repelems(selectionsAvg,[1;17])' repelems(weightedTruncationAvg,[1;17])'];

TempTable = [num2cell(OutputMatrix) key' repmat({"Test"},17,1) repmat(alg(algorithm),17,1) repmat({num2str(tol)},17,1)];

outputCell = [outputCell;TempTable];

clearvars -except choice1 algorithm choice4 choice0 singularvectors outputCell  trainortest alg bestSV tol row col param
end
clearvars -except choice1 algorithm choice4 choice0 singularvectors outputCell  trainortest alg bestSV tol row col param
end

TableHeaders = {"subset" "fullset" "SVrank" "max selections" "avg selections" "avg truncation" "annotations" "data set" "algorithm" "tol"};

OutputTable = [TableHeaders; outputCell];

savename = strcat("MITBIH_Test.csv")
cell2csv(savename,OutputTable);
