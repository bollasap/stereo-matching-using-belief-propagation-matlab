dispLevels = 16;
iterations = 80;
lambda = 5;
threshold = dispLevels-1; %don't use threshold

% Set the disparity values
d = 0:dispLevels-1;

% Read the stereo image and convert it to grayscale
left = rgb2gray(imread('Left.png'));
right = rgb2gray(imread('Right.png'));

% Apply a Gaussian filter
left = imgaussfilt(left,0.6,'FilterSize',5);
right = imgaussfilt(right,0.6,'FilterSize',5);

% Get the image size
[height,width] = size(left);

% Compute data term
dataTerm = zeros(height,width,dispLevels);
for i = 1:dispLevels
	right_d = [zeros(height,d(i)),right(:,1:end-d(i))];
	dataTerm(:,:,i) = abs(double(left)-double(right_d));
end

% Compute smoothness term
smoothnessTerm = lambda*min(abs(d-d'),threshold);

% Initialize messages
msgUp = zeros(height,width,dispLevels);
msgDown = zeros(height,width,dispLevels);
msgRight = zeros(height,width,dispLevels);
msgLeft = zeros(height,width,dispLevels);

figure

% Start iterations
for i = 1:iterations
	% Auxiliary tables that help us create the messages
	U = dataTerm + msgDown + msgRight + msgLeft;
	D = dataTerm + msgUp + msgRight + msgLeft;
	R = dataTerm + msgUp + msgDown + msgLeft;
	L = dataTerm + msgUp + msgDown + msgRight;
	
	% For each pixel
	for y = 2:height-1
		for x = 2:width-1
			% Send message up
			msg = reshape(U(y,x,:),[dispLevels,1]);
			msg = min(msg+smoothnessTerm);
			msg = msg-min(msg);
			msgDown(y-1,x,:) = msg;
			
			% Send message down
			msg = reshape(D(y,x,:),[dispLevels,1]);
			msg = min(msg+smoothnessTerm);
			msg = msg-min(msg);
			msgUp(y+1,x,:) = msg;
			
			% Send message right
			msg = reshape(R(y,x,:),[dispLevels,1]);
			msg = min(msg+smoothnessTerm);
			msg = msg-min(msg);
			msgLeft(y,x+1,:) = msg;
			
			% Send message left
			msg = reshape(L(y,x,:),[dispLevels,1]);
			msg = min(msg+smoothnessTerm);
			msg = msg-min(msg);
			msgRight(y,x-1,:) = msg;
		end
	end
	
	% Compute belief
	belief = dataTerm + msgUp + msgDown + msgRight + msgLeft;
	
	% Update disparity map
	[Y,I] = min(belief,[],3);
	dispMap = d(I);
	
	% Update disparity image
	scaleFactor = 256/dispLevels;
	dispImage = uint8(dispMap*scaleFactor);
	
	% Show disparity image
	imshow(dispImage)
	
	% Show current iteration
	fprintf('iteration %d/%d\n',i,iterations)
end

% Save disparity image
imwrite(dispImage,'Disparity.png')