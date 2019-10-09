function delete_button(index){
    $.post('/delete_run',
    {'index': index},
    function(data){
        $( "#" + index ).remove();
        adjust_table_row_ids();
    });
}

function adjust_table_row_ids(index){
    var rowCount = $( "#status_table tr" ).length;
    for (var i=1; i <= rowCount; i++){
        if (i >= index){
            var alt_i = i + 1
        } else{
            var alt_i = i
        }
        $( "#status_table tr" ).eq(alt_i).attr('id', i)
        $( "#status_table tr" ).eq(alt_i).children().each(function () {
            if (this.id !== ""){
                this.id = this.id.replace(/\d+$/, "") + i;
            }
            if (this.querySelector("a") !== null && this.querySelector("a").onclick !== null){
                sig = get_func_signature(this.querySelector("a").onclick);
                this.querySelector("a").setAttribute( "onClick", sig + "(" + i + ")");
            }
        });
    }
}

function get_func_signature(func) {
    var f = func.toString();
    var a = f.match(/[{][\n](.+)[(](.+)[)][\n][}]/i);

    return a[1];
}

function refresh_status(index){
    $.post('/refresh_status',
    {'index': index},
    function(data){
        $( "#update_time" + index ).html(data['update_time']);
        $( "#status" + index ).html(data['status']);
        if (data['status'].includes('Data available')){
        	$( "#download_button" + index).contents().filter(function () { return this.nodeType === 3; }).remove();
            $( "#download_button" + index + " button").replaceWith('<button type="submit"  name="index" value="' + index + '" class="link-button">Download Data</button>');

            $( "#gen_more_data_button" + index).contents().filter(function () { return this.nodeType === 3; }).remove();
            $( "#gen_more_data_button" + index + " button" ).replaceWith('<button type="submit"  name="index" value="' + index + '" class="link-button">Generate More Data</button>');

			$( "#visualize_button" + index).contents().filter(function () { return this.nodeType === 3; }).remove();
            $( "#visualize_button" + index + " button" ).replaceWith('<button type="submit"  name="index" value="' + index + '" class="link-button">See Visualizations</button>');

			$( "#continue_training_button" + index).contents().filter(function () { return this.nodeType === 3; }).remove();
            $( "#continue_training_button" + index + " button" ).replaceWith('<button type="submit"  name="index" value="' + index + '" class="link-button">Train Longer</button>');
        }
    });
}

function download_data(index){
    $.post('/download_data',
    {'index': index},
    function(data){

    });
}
