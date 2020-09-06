input1=rand(3,2);
inputreal=input1;
clf = MinMaxScaler();
(clf.fit(input1));
input1=(clf.transform(input1));
predicted=clf.inverse_transform(input1);