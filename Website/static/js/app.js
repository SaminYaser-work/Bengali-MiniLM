var data = [];
var token = "";

jQuery(document).ready(function () {
  $("#input_text").on("keyup", function (e) {
    // স্পেস বোতাম চাপলে পরবর্তী শব্দের প্রস্তাবগুলো দেখানো হবে
    if (e.key == " ") {
      $.ajax({
        url: "/get_end_predictions",
        type: "post",
        contentType: "application/json",
        dataType: "json",
        data: JSON.stringify({
          input_text: $("#input_text").val(),
          //   top_k: slider.val(),
        }),
        beforeSend: function () {
          $(".overlay").show();
        },
        complete: function () {
          $(".overlay").hide();
        },
      })
        .done(function (jsondata, textStatus, jqXHR) {
          console.log(jsondata);
          $("#text_our").val(jsondata["our"]);
          $("#text_bb").val(jsondata["bb"]);
          $("#text_bb2").val(jsondata["bb2"]);
          $("#text_xlm").val(jsondata["xlm"]);
          $("#text_minilm").val(jsondata["minilm"]);
          $("#text_bert").val(jsondata["bert"]);
        })
        .fail(function (jsondata, textStatus, jqXHR) {
          console.log(jsondata);
        });
    }
  });

  // শব্দ পূরণ এর জন্য
  $("#btn-process").on("click", function () {
    $.ajax({
      url: "/get_mask_predictions",
      type: "post",
      contentType: "application/json",
      dataType: "json",
      data: JSON.stringify({
        input_text: $("#mask_input_text").val(),
        // top_k: slider_mask.val(),
      }),
      beforeSend: function () {
        $(".overlay").show();
      },
      complete: function () {
        $(".overlay").hide();
      },
    })
      .done(function (jsondata, textStatus, jqXHR) {
        console.log(jsondata);
        $("#mask_text_our").val(jsondata["our"]);
        $("#mask_text_bb").val(jsondata["bb"]);
        $("#mask_text_bb2").val(jsondata["bb2"]);
        $("#mask_text_xlm").val(jsondata["xlm"]);
        $("#mask_text_minilm").val(jsondata["minilm"]);
        $("#mask_text_bert").val(jsondata["bert"]);
      })
      .fail(function (jsondata, textStatus, jqXHR) {
        console.log(jsondata);
      });
  });
});
