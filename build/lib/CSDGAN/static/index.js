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
        if (data['status'].includes('Data available') || data['status'].includes('Error') ){
            $( "#download_button" + index).contents().filter(function () { return this.nodeType === 3; }).remove();
            $( "#download_button" + index + " button").replaceWith('<button type="submit"  name="index" value="' + index + '" class="link-button">Download Data</button>');

            $( "#gen_more_data_button" + index).contents().filter(function () { return this.nodeType === 3; }).remove();
            $( "#gen_more_data_button" + index + " button" ).replaceWith('<button type="submit"  name="index" value="' + index + '" class="link-button">Generate More Data</button>');

            $( "#delete_button" + index).contents().filter(function () { return this.nodeType === 3; }).remove();
            $( "#delete_button" + index + " a" ).replaceWith('<a href="#" onclick="delete_button(' + index + ')">Delete</a>');
        }
    });
}

function download_data(index){
    $.post('/download_data',
    {'index': index},
    function(data){

    });
}
