using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;

namespace TextToMotionWeb.Controllers
{
    public class TextToMotionController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Search(string Query){
            ViewData["Title"] = "Search Results";
            ViewData["Query"] = Query;   
            return View();
        }

        [HttpPost]
        public action Search()
        {
            string searchQuery = Request.Form["Search"];

        }
    }
}