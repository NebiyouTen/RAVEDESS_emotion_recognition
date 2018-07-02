from bottle import Bottle

app = Bottle()

@app.route('/')
def index():
    return '''
            <form>
            <h1>Record Audio</h1>
            <button id="record">Record</button>
             <button class="stop">Stop</button>
            <script>
	navigator.getUserMedia = ( navigator.getUserMedia ||
						   navigator.webkitGetUserMedia ||
						   navigator.mozGetUserMedia ||
						   navigator.msGetUserMedia);
	var audioCtx = new (window.AudioContext || webkitAudioContext)();
	if (navigator.getUserMedia) {
	   console.log('getUserMedia supported.');
	   navigator.getUserMedia (
		  {
			 audio: true
		  },
	 );
	} 


	 $('#record').click(function(){
			 var mediaRecorder = new MediaRecorder(stream);
			 visualize(stream);
			 record.onclick = function() {
				mediaRecorder.start();
				console.log(mediaRecorder.state);
				console.log("recorder started");
				record.style.background = "red";
				record.style.color = "black";
			 }

			 stop.onclick = function() {
				mediaRecorder.stop();
			  console.log(mediaRecorder.state);
				console.log("recorder stopped");
				record.style.background = "";
				record.style.color = "";
			 }
	});
</script>
            </form>

    '''
   
@app.route('/audio',method='POST')
def about():
    return "AUdio UPloaded"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
    
