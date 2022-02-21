%first, the audio is read.
[s,fs] = audioread('polyushka.wav');
% Audio is resampled at 16000/fs times the original sample rate which is obviously equal to 16000
s = resample(s,16000,fs);

%by using stft function, the spectrum of B is extracted and plotted
% 2048 is fft size, 256 is hopsize between adjacent frames, 0 is padding
% and hann(2048) is window
spectrum = stft(s', 2048, 256, 0, hann(2048));
figure; plot(spectrum); title ('originalspectrum');

%the magnitude and the phase of audio is calculated and plotted
music = abs(spectrum);
figure; plot(music); title ('original spectrum magnitude');
sphase = spectrum ./ (abs(spectrum) + eps); %eps is added in case there is 0 in spectrum
figure; plot(sphase); title ('original spectrum phase');

%read all the .wav files in the following directory
notesfolder = 'notes15/';
listname = dir([notesfolder '*.wav']);


%define empty array for notes
notes = [];

%create a loop that will save the spectrums of each note / .wav file to the
%notes variable as different columns
for k = 1:length(listname)
    [n,fn] = audioread([notesfolder listname(k).name]);
    n = n(:,1);
    n = resample(n,16000,fn);
    spectrum_n = stft(n', 2048, 256, 0, hann(2048));
    %find the central frame
    middle = ceil(size(spectrum_n,2)/2);
    note = abs(spectrum_n(:,middle));
    %clean up everything more than 40 db below the peak
    note(find(note < max(note(:))/100)) = 0;
    %normalise the note to unit length
    note = note / norm(note);
    % assign the calculated note to the empty array
    notes = [notes, note];
end

%since the audio is composed of these notes, (notes * w = music) where w is
%the ith row of w is the transcription of the ith note. We calculate and plot w.

w = pinv(notes) * music; %we use pinv() instead of inv() since notes is not square matrix
figure; plot(w); title ('W');

%all negative values in w is converted to 0 since there cannot be
%negative magnitude for a frequency
[i1,i2] = size(w);
for i=1:i1*i2
   if w(i)<=0
       w(i)=0;
   end
end

%the audio spectrum is reconstructed and plotted using the original formula (notes * w)
reconstructed = notes * w;
figure; plot(reconstructed); title ('reconstructed spectrum');

%the reconstructed spectrum is then converted to reconstructed signal with 
%stft function, with the same phase, fft size, hopsize, padding and window   
rsignal = stft(reconstructed.*sphase,2048,256,0,hann(2048));

%the player variable of the audio is defined using audioplayer function
player = audioplayer(rsignal, fs);

%the audio is played
play(player);
