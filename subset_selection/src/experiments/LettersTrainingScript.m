pkg load io
outputCell = [];

load train_letter_recognition_data.mat

A = feature_matrix;
B = A;

tol = 1e-10;

for algorithm = 1:22
algorithm
for singularvectors = 2:2:16
singularvectors

alg = {"EDEIM", "LDEIM", "INF_LDEIM", "ONE_LDEIM", "DEIMQR", "QDEIM_EDEIM", "LEVERAGE_EDEIM", "QR_EDEIM", "OneLEVERAGE_EDEIM", "InfLEVERAGE_EDEIM", "QR","DEIM","QDEIM","DEIMCSDEIM","QDEIMCSDEIM","QRCSQR","DEIMCSQR","QDEIMCSQR","QRCSDEIM","LeverageScore","RRQR mem_l2","FRQR mem_l2"};

binarycount = zeros(1,27);
key = {"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","Error"};
output = zeros(1,27);
subsetsum = zeros(1,27);
fullsetsum = zeros(1,27);
selectionsLimit = 32;

    [U,S,V] = svd(full(B),'econ');
    lettersfullset = condense_letter(labels);
    lettersfullset(lettersfullset ~= 0) = 1;
    fullsetsum = fullsetsum + lettersfullset;
    if algorithm == 1 #EDEIM
        [p] = NewEDeim(U(:,1:singularvectors),'mem_coh',tol,selectionsLimit);
    elseif algorithm == 2 #LDEIM
        [p] = LDEIMv4(U(:,1:singularvectors),selectionsLimit);
    elseif algorithm == 3 #INF_LDEIM
        [p] = ModifiedLDEIM(U(:,1:singularvectors),selectionsLimit,inf);
    elseif algorithm == 4 #ONE_LDEIM
        [p] = ModifiedLDEIM(U(:,1:singularvectors),selectionsLimit,1);
    elseif (algorithm == 5) #DEIMQR
        [p] = DEIMQR(U(:,1:singularvectors),selectionsLimit);
    elseif algorithm == 6 #QDEIM_EDEIM
        [p] = QDEIMandRestartedDEIMv2(U(:,1:singularvectors),'mem_coh',tol,selectionsLimit);
    elseif algorithm == 7 #LEVERAGE_EDEIM
        [p] = LeverageandRestartedDEIMv2(U(:,1:singularvectors),'mem_coh',tol,selectionsLimit,2);
    elseif algorithm == 8 %QR EDEIM
        [~,~,p] = qr(B', 'vector');
        limit = min([selectionsLimit, singularvectors]);
        p = p(1:limit)';
        p = RestartedDEIM(p,U(:,1:singularvectors),'mem_coh',tol,selectionsLimit);
    elseif algorithm == 9 #OneLEVERAGE_EDEIM
        [p] = LeverageandRestartedDEIM(U(:,1:singularvectors),'mem_coh',tol,selectionsLimit,1);
    elseif algorithm == 10 #InfLEVERAGE_EDEIM
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
    elseif algorithm == 21 %RRQR mem_l2
        p = RRQR(U(:,1:singularvectors), selectionsLimit, 1e-10, 'mem_l2');
    elseif algorithm == 22 %FRQR mem_l2
        [p,~] = FRQR(U(:,1:singularvectors), selectionsLimit, 'mem_l2');
    end
    letterssubset = condense_letter(labels(p));
    letterssubset(letterssubset ~= 0) = 1;
    subsetsum = subsetsum + letterssubset;
    sizeOfP = max(size(p));

OutputMatrix = [subsetsum' fullsetsum' repelems(singularvectors,[1;27])' repelems(selectionsLimit,[1;27])' repelems(sizeOfP,[1;27])'];

TempTable = [num2cell(OutputMatrix) key' repmat({"Train"},27,1) repmat(alg(algorithm),27,1) repmat({num2str(tol)},27,1)];

outputCell = [outputCell;TempTable];

clearvars -except choice1 algorithm choice4 choice0 singularvectors outputCell trainortest alg bestSV tol B A feature_matrix labels parameters param col row
end
clearvars -except choice1 algorithm choice4 choice0 singularvectors outputCell trainortest alg bestSV tol B A feature_matrix labels parameters param col row
end

TableHeaders = {"subset" "fullset" "SVrank" "max selections" "sizeOfP" "annotations" "data set" "algorithm" "tol"};

OutputTable = [TableHeaders; outputCell];

%savename = input("FileName?")

savename = strcat("Letters_Train.csv")
cell2csv(savename,OutputTable);
