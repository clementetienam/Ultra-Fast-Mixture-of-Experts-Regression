function A= Read_data(data)
Tbl = readtable(data);
%Tbl=rmmissing(Tbl); 
VarNames = Tbl.Properties.VariableNames;                                 % Recover Variable Names (Column Titles)
VarInputs=cell(5,1);
VarInputs{1,1}='BDFCRG';
VarInputs{2,1}='HQCRG';
VarInputs{3,1}='SCCRG';
VarInputs{4,1}='BDFADD';
VarInputs{5,1}='HQADD';

for ii=1:5
ColIdx = find(strcmp(VarNames, VarInputs{ii,1})); 
use=Tbl(:,ColIdx);
use2=use{:,:};
Inns(:,ii)=use2;
end

%% 
Varoutputs=cell(7,1);
Varoutputs{1,1}='BlackSpeck__SFL_';
Varoutputs{2,1}='ColourAValue';
Varoutputs{3,1}='ColourBValue';
Varoutputs{4,1}='ColourLValue';
Varoutputs{5,1}='DPSContent';
Varoutputs{6,1}='MeltViscosity';
Varoutputs{7,1}='MoistureContentP';

for ij=1:7
ColIdx = find(strcmp(VarNames, Varoutputs{ij,1})); 
usea=Tbl(:,ColIdx);
useb1=usea{:,:};
outts(:,ij)=useb1;
end

A=[Inns outts];
A=rmmissing(A); 
end