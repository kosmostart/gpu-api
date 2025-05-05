use tokio::runtime::Runtime;
use warp::{Filter, http::Response};

fn main() {
    env_logger::init();
    let rt = Runtime::new().expect("Failed to create runtime");

    let app_name = "hi";

    let routes = 
        warp::path(app_name).and(warp::path::end())
            .map(|| Response::builder().body(index(app_name, "test-app")))
        .or(warp::path(app_name).and(warp::fs::dir("../")));
    
    rt.block_on(warp::serve(routes).run(([127, 0, 0, 1], 12345)));
}

fn index(app_name: &str, app_file_name: &str) -> String {        
    let js_path = "/".to_owned() + app_name + "/web/" + app_file_name + ".js";
    let wasm_path = "/".to_owned() + app_name + "/web/" + app_file_name +"_bg.wasm";	

	format!(r#"
    <!doctype html>    

    <head>
        <meta charset="utf-8" />
        <title>sky-gui</title>        
    </head>

    <body>        
        <script type="module">
            import init from "{}";

            init("{}");
        </script>
    </body>

    </html>
    "#, js_path, wasm_path)
}