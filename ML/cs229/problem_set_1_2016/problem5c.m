

% smooth the quasars with local linear regression

train_smooth = train_qso;
test_smooth = test_qso;
tau = 5;
%add biase term 
X = [ones(nn,1) lambdas];
y = transpose(train_qso(1,:));

% smooth all quasars
for jj = 1:mm
	ytrain = transpose(train_qso(jj,:));
	train_smooth(jj,:) = transpose(local_linear_regression(lambdas,ytrain,tau));
end

%for jj = 1:mtest
%	ytest = transpose(test_qso(jj,:));
%	test_smooth(jj,:) = transpose(local_linear_regression(lambdas,ytest,tau));
%end

%find the right most funcyion parts , the one to the right of lyman-alpha 

right_trains = train_smooth(:,151:end);
left_trains = train_smooth(:,1:50);

%%construct matrix of all pairs of distances between training quasar spectra 
train_dist = zeros(mm,mm);
for ii = 1:mm
	for jj = (ii+1):mm
		train_dist(ii,jj) = norm(right_trains(ii,:)-right_trains(jj,:))^2;
	end
end

train_dist = train_dist + transpose(train_dist);
train_dist = train_dist/max(train_dist(:));

f_left_estimators = zeros(mm,50);
num_nearest = 3;

for ii= 1:mm 
	[train_dist_sort, inds] = sort(train_dist(ii,:),1,'ascend');
	close_inds = ones(mm,1);
	close_inds(inds((num_nearest+1):end)) = 0;
	h = max(train_dist(:,ii));
	kerns = max(1-train_dist(:,ii)/h,0);
	kerns = kerns.*close_inds;
	f_left_estimators(ii,:) = transpose(left_trains) *kerns/sum(kerns);
end

plot(lambdas(1:50),f_left_estimators(1,:),'r-','linewidth',2)