pkg load io


outputCell = [];

load SoybeanTesting.mat

A = data;
B = A;

## For tagging the results table
alg = {"EDEIM", "LDEIM", "INF_LDEIM", "ONE_LDEIM", "DEIMQR", "QDEIM_EDEIM", "LEVERAGE_EDEIM", "QR_EDEIM", "OneLEVERAGE_EDEIM", "InfLEVERAGE_EDEIM", "QR","DEIM","QDEIM","DEIMCSDEIM","QDEIMCSDEIM","QRCSQR","DEIMCSQR","QDEIMCSQR","QRCSDEIM","LeverageScore"};

param = [1 2 6 10; 4 2 10 12 ; 17 2 10 16; 15 2 14 16; 8 2 10 14; 20 2 8 16]
for row = 1:6
algorithm = param(row,1)
for col = 1:3
singularvectors = param(row, 1+col)

key = {"alternarialeaf-spot","anthracnose","bacterial-blight","bacterial-pustule","brown-spot","brown-stem-rot","charcoal-rot","diaporthe-stem-canker","downy-mildew","frog-eye-leaf-spot","phyllosticta-leaf-spot","phytophthora-rot","powdery-mildew","purple-seed-stain","rhizoctonia-root-rot","Error"};

binarycount = zeros(1,16);
output = zeros(1,16);
subsetsum = zeros(1,16);
fullsetsum = zeros(1,16);
selectionsLimit = 30;

tol = 1e-10;

    [U,S,V] = svd(full(B),'econ');
    soybeanfullset = condense_soybean(annotations);
    soybeanfullset(soybeanfullset ~= 0) = 1;
    fullsetsum = fullsetsum + soybeanfullset;
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
    end
    soybeansubset = condense_soybean(annotations(p));
    soybeansubset(soybeansubset ~= 0) = 1;
    subsetsum = subsetsum + soybeansubset;
    sizeOfP = max(size(p));

OutputMatrix = [subsetsum' fullsetsum' repelems(singularvectors,[1;16])' repelems(sizeOfP,[1;16])'];

TempTable = [num2cell(OutputMatrix) key' repmat({"Test"},16,1) repmat(alg(algorithm),16,1) repmat({num2str(tol)},16,1)];

outputCell = [outputCell;TempTable];

clearvars -except choice1 algorithm choice4 choice0 singularvectors outputCell trainortest alg bestSV tol B A data annotations parameters param row col
end
clearvars -except choice1 algorithm choice4 choice0 singularvectors outputCell trainortest alg bestSV tol B A data annotations parameters param row col
end
TableHeaders = {"subset" "fullset" "SVrank" "sizeOfP" "annotations" "data set" "algorithm" "tol"};

OutputTable = [TableHeaders; outputCell];

%savename = input("FileName?")

savename = strcat("Soybean_Test.csv")
cell2csv(savename,OutputTable);
